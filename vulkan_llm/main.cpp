#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <cstring>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <dirent.h>
#include "vulkan_compute.hpp"

static std::atomic<bool> training_interrupted{false};
extern "C" void handle_sigint(int) {
    training_interrupted = true;
}

static double check_ram() {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (double)pages * page_size / (1024.0 * 1024.0);
}

static std::string format_eta(float secs);
static std::string rgb_fg(int r, int g, int b);
static float loss_to_t(float loss);
static float eta_to_t(float eta);
static std::string gradient_color(float t);
static std::string star_color(float t);
static std::string hash_color(float t);
static std::string epoch_color(int epoch, int max_epochs);
static std::string rgb_fg(int r, int g, int b);
static float loss_to_t(float loss);
static float eta_to_t(float eta);
static std::string gradient_color(float t);
static std::string bar_color(float t);

static void backup_model(const std::string& path);
static std::string read_backup_dir_from_file(const std::string& path);

struct Tokenizer {
    std::map<char, int> stoi;
    std::map<int, char> itos;
    int vocab_size;

    Tokenizer() : vocab_size(0) {}

    void build(const std::string& t) {
        std::string chars = t;
        std::sort(chars.begin(), chars.end());
        chars.erase(std::unique(chars.begin(), chars.end()), chars.end());
        vocab_size = chars.size();
        stoi.clear();
        itos.clear();
        for (int i = 0; i < vocab_size; i++) {
            stoi[chars[i]] = i;
            itos[i] = chars[i];
        }
    }
};

struct NeuralLM {
    VulkanCompute* vk;
    bool use_cpu;
    int V, H;
    float lr;
    int batch_size;
    float temperature;
    float *W1, *b1, *W2, *b2;

    VkBuffer gpu_in, gpu_hidden, gpu_logits, gpu_W1, gpu_W2, gpu_b2, gpu_b1;
    VkDeviceMemory mem_in, mem_hidden, mem_logits, mem_W1, mem_W2, mem_b2, mem_b1;
    float *ptr_in, *ptr_hidden, *ptr_logits;
    float *ptr_W1, *ptr_W2, *ptr_b1, *ptr_b2;

    VkBuffer gpu_inputs, gpu_targets, gpu_hidden_batch, gpu_logits_batch;
    VkDeviceMemory mem_inputs, mem_targets, mem_hidden_batch, mem_logits_batch;
    VkBuffer gpu_dW1, gpu_dW2, gpu_db1, gpu_db2, gpu_losses;
    VkDeviceMemory mem_dW1, mem_dW2, mem_db1, mem_db2, mem_losses;
    int *ptr_inputs_int, *ptr_targets_int;
    float *ptr_losses;
    float *ptr_dW1, *ptr_dW2, *ptr_db1, *ptr_db2;

    int top_k;
    std::string backup_dir;

    static std::string generate_backup_dir() {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        unsigned seed = (unsigned)t ^ (unsigned)(uintptr_t)&t;
        std::srand(seed);
        std::stringstream ss;
        ss << "ModelBackups/" << t << "_" << std::hex << std::rand() << "/backups";
        return ss.str();
    }

    NeuralLM(VulkanCompute& _vk, int vocab_size, int _H, float _lr, int _batch, float _temp,
             int _top_k = 20, bool _cpu = false)
        : vk(_cpu ? nullptr : &_vk), use_cpu(_cpu), V(vocab_size), H(_H), lr(_lr),
          batch_size(_batch), temperature(_temp), top_k(_top_k) {
        backup_dir = generate_backup_dir();
        W1 = new float[V * H]();
        b1 = new float[H]();
        W2 = new float[H * V]();
        b2 = new float[V]();

        std::mt19937 gen(42);
        float h_scale = sqrtf(2.0f / V);
        for (int i = 0; i < V * H; i++) W1[i] = ((float)gen() / gen.max() * 2 - 1) * h_scale;
        for (int i = 0; i < H * V; i++) W2[i] = ((float)gen() / gen.max() * 2 - 1) * 0.01f;

        if (use_cpu) {
            ptr_in = new float[V]();
            ptr_hidden = new float[H]();
            ptr_logits = new float[V]();
            ptr_W1 = W1; ptr_W2 = W2; ptr_b1 = b1; ptr_b2 = b2;
            ptr_inputs_int = new int[batch_size]();
            ptr_targets_int = new int[batch_size]();
            ptr_losses = new float[batch_size]();
            ptr_dW1 = new float[batch_size * H]();
            ptr_dW2 = new float[batch_size * H * V]();
            ptr_db1 = new float[batch_size * H]();
            ptr_db2 = new float[batch_size * V]();
        } else {
            vk->createBuffer(V * sizeof(float), gpu_in, mem_in);
            vk->createBuffer(H * sizeof(float), gpu_hidden, mem_hidden);
            vk->createBuffer(V * sizeof(float), gpu_logits, mem_logits);
            vk->createBuffer(V * H * sizeof(float), gpu_W1, mem_W1);
            vk->createBuffer(H * V * sizeof(float), gpu_W2, mem_W2);
            vk->createBuffer(V * sizeof(float), gpu_b2, mem_b2);
            vk->createBuffer(H * sizeof(float), gpu_b1, mem_b1);

            size_t maxB = batch_size;
            vk->createBuffer(maxB * sizeof(int), gpu_inputs, mem_inputs);
            vk->createBuffer(maxB * sizeof(int), gpu_targets, mem_targets);
            vk->createBuffer(maxB * H * sizeof(float), gpu_hidden_batch, mem_hidden_batch);
            vk->createBuffer(maxB * V * sizeof(float), gpu_logits_batch, mem_logits_batch);
            vk->createBuffer(maxB * H * sizeof(float), gpu_dW1, mem_dW1);
            vk->createBuffer(maxB * H * V * sizeof(float), gpu_dW2, mem_dW2);
            vk->createBuffer(maxB * H * sizeof(float), gpu_db1, mem_db1);
            vk->createBuffer(maxB * V * sizeof(float), gpu_db2, mem_db2);
            vk->createBuffer(maxB * sizeof(float), gpu_losses, mem_losses);

            vkMapMemory(vk->device, mem_in, 0, V * sizeof(float), 0, (void**)&ptr_in);
            vkMapMemory(vk->device, mem_hidden, 0, H * sizeof(float), 0, (void**)&ptr_hidden);
            vkMapMemory(vk->device, mem_logits, 0, V * sizeof(float), 0, (void**)&ptr_logits);
            vkMapMemory(vk->device, mem_inputs, 0, maxB * sizeof(int), 0, (void**)&ptr_inputs_int);
            vkMapMemory(vk->device, mem_targets, 0, maxB * sizeof(int), 0, (void**)&ptr_targets_int);
            vkMapMemory(vk->device, mem_losses, 0, maxB * sizeof(float), 0, (void**)&ptr_losses);
            vkMapMemory(vk->device, mem_W1, 0, V * H * sizeof(float), 0, (void**)&ptr_W1);
            vkMapMemory(vk->device, mem_W2, 0, H * V * sizeof(float), 0, (void**)&ptr_W2);
            vkMapMemory(vk->device, mem_b1, 0, H * sizeof(float), 0, (void**)&ptr_b1);
            vkMapMemory(vk->device, mem_b2, 0, V * sizeof(float), 0, (void**)&ptr_b2);
            vkMapMemory(vk->device, mem_dW1, 0, maxB * H * sizeof(float), 0, (void**)&ptr_dW1);
            vkMapMemory(vk->device, mem_dW2, 0, maxB * H * V * sizeof(float), 0, (void**)&ptr_dW2);
            vkMapMemory(vk->device, mem_db1, 0, maxB * H * sizeof(float), 0, (void**)&ptr_db1);
            vkMapMemory(vk->device, mem_db2, 0, maxB * V * sizeof(float), 0, (void**)&ptr_db2);

            sync_weights_gpu();
        }
    }

    void destroy_buffers(VulkanCompute& _vk) {
        if (use_cpu) {
            delete[] ptr_in; delete[] ptr_hidden; delete[] ptr_logits;
            delete[] ptr_inputs_int; delete[] ptr_targets_int; delete[] ptr_losses;
            delete[] ptr_dW1; delete[] ptr_dW2; delete[] ptr_db1; delete[] ptr_db2;
            return;
        }
        auto free = [&](VkBuffer b, VkDeviceMemory m) {
            vkDestroyBuffer(_vk.device, b, nullptr);
            vkFreeMemory(_vk.device, m, nullptr);
        };
        free(gpu_in, mem_in); free(gpu_hidden, mem_hidden); free(gpu_logits, mem_logits);
        free(gpu_W1, mem_W1); free(gpu_W2, mem_W2); free(gpu_b1, mem_b1); free(gpu_b2, mem_b2);
        free(gpu_inputs, mem_inputs); free(gpu_targets, mem_targets);
        free(gpu_hidden_batch, mem_hidden_batch); free(gpu_logits_batch, mem_logits_batch);
        free(gpu_dW1, mem_dW1); free(gpu_dW2, mem_dW2);
        free(gpu_db1, mem_db1); free(gpu_db2, mem_db2);
        free(gpu_losses, mem_losses);
    }

    void sync_weights_gpu() {
        if (!use_cpu) {
            memcpy(ptr_W1, W1, V * H * sizeof(float));
            memcpy(ptr_W2, W2, H * V * sizeof(float));
            memcpy(ptr_b1, b1, H * sizeof(float));
            memcpy(ptr_b2, b2, V * sizeof(float));
        }
    }

    void sync_weights_cpu() {
        if (!use_cpu) {
            memcpy(W1, ptr_W1, V * H * sizeof(float));
            memcpy(W2, ptr_W2, H * V * sizeof(float));
            memcpy(b1, ptr_b1, H * sizeof(float));
            memcpy(b2, ptr_b2, V * sizeof(float));
        }
    }

    void forward_single(int input_char, float* logits_out) {
        if (use_cpu) {
            for (int j = 0; j < H; j++) {
                float s = b1[j] + W1[input_char * H + j];
                ptr_hidden[j] = s > 0 ? s : 0;
            }
            for (int j = 0; j < V; j++) {
                float s = b2[j];
                for (int k = 0; k < H; k++)
                    s += ptr_hidden[k] * W2[k * V + j];
                logits_out[j] = s;
            }
        } else {
            for (int j = 0; j < V; j++) ptr_in[j] = 0.0f;
            ptr_in[input_char] = 1.0f;
            vk->executeMatMul(gpu_in, gpu_W1, gpu_hidden, 1, V, H);
            for (int j = 0; j < H; j++) ptr_hidden[j] = std::max(0.0f, ptr_hidden[j] + b1[j]);
            vk->executeMatMul(gpu_hidden, gpu_W2, gpu_logits, 1, H, V);
            for (int j = 0; j < V; j++) logits_out[j] = ptr_logits[j] + b2[j];
        }
    }

    void train_batch_cpu(const std::vector<int>& data, size_t start, int B,
                         float& total_loss, int& count) {
        for (int i = 0; i < B; i++) {
            int inp = data[start + i];
            int tgt = data[start + i + 1];
            ptr_inputs_int[i] = inp;
            ptr_targets_int[i] = tgt;
        }
        for (int i = 0; i < B * H; i++) ptr_dW1[i] = 0;
        for (int i = 0; i < B * H * V; i++) ptr_dW2[i] = 0;
        for (int i = 0; i < B * H; i++) ptr_db1[i] = 0;
        for (int i = 0; i < B * V; i++) ptr_db2[i] = 0;

        for (int i = 0; i < B; i++) {
            int inp = ptr_inputs_int[i];
            int tgt = ptr_targets_int[i];
            float* hidden = ptr_hidden;
            for (int j = 0; j < H; j++) {
                float s = b1[j] + W1[inp * H + j];
                hidden[j] = s > 0 ? s : 0;
            }
            float maxl = -1e38f;
            for (int j = 0; j < V; j++) {
                float s = b2[j];
                for (int k = 0; k < H; k++)
                    s += hidden[k] * W2[k * V + j];
                ptr_logits[j] = s;
                if (s > maxl) maxl = s;
            }
            float sum_exp = 0;
            for (int j = 0; j < V; j++) {
                float e = expf(ptr_logits[j] - maxl);
                ptr_in[j] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;
            float loss = -logf(ptr_in[tgt] * inv_sum + 1e-10f);
            ptr_losses[i] = loss;
            total_loss += loss;
            count++;

            float* dlogits = ptr_logits;
            for (int j = 0; j < V; j++)
                dlogits[j] = ptr_in[j] * inv_sum - (j == tgt ? 1.0f : 0.0f);

            for (int j = 0; j < V; j++) {
                for (int k = 0; k < H; k++)
                    ptr_dW2[i * H * V + k * V + j] += hidden[k] * dlogits[j];
                ptr_db2[i * V + j] += dlogits[j];
            }
            for (int k = 0; k < H; k++) {
                float dh = 0;
                for (int j = 0; j < V; j++)
                    dh += dlogits[j] * W2[k * V + j];
                dh *= hidden[k] > 0 ? 1.0f : 0.0f;
                ptr_dW1[i * H + k] += (float)inp >= 0 ? W1[inp * H + k] * 0 : dh;
                ptr_db1[i * H + k] += dh;
                for (int r = 0; r < V; r++)
                    ptr_dW1[i * H + k] += (r == inp ? 1.0f : 0.0f) * dh;
            }
        }
    }

    void reduce_update(int B) {
        int w1s = V * H;
        int w2s = H * V;
        int b1s = H;
        float invB = 1.0f / B;
        for (int id = 0; id < w1s + w2s + b1s + V; id++) {
            if (id < w1s) {
                int r = id / H, c = id % H;
                float sum = 0;
                for (int i = 0; i < B; i++)
                    if (ptr_inputs_int[i] == r)
                        sum += ptr_dW1[i * H + c];
                W1[r * H + c] -= lr * sum * invB;
            } else if (id < w1s + w2s) {
                int id2 = id - w1s;
                int r = id2 / V, c = id2 % V;
                float sum = 0;
                for (int i = 0; i < B; i++)
                    sum += ptr_dW2[i * H * V + r * V + c];
                W2[r * V + c] -= lr * sum * invB;
            } else if (id < w1s + w2s + b1s) {
                int id2 = id - w1s - w2s;
                float sum = 0;
                for (int i = 0; i < B; i++)
                    sum += ptr_db1[i * H + id2];
                b1[id2] -= lr * sum * invB;
            } else {
                int id2 = id - w1s - w2s - b1s;
                float sum = 0;
                for (int i = 0; i < B; i++)
                    sum += ptr_db2[i * V + id2];
                b2[id2] -= lr * sum * invB;
            }
        }
    }

    void train_hybrid(const std::vector<int>& data, int num_cores, std::atomic<bool>& interrupt,
                      int max_epochs = 0, const std::string& save_path = "", bool save_every_epoch = false) {
        std::cout << "Training (V=" << V << ", H=" << H
                  << ", data=" << data.size() << ", batch=" << batch_size
                  << ", cores=" << num_cores << ")...\n" << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();
        auto t_batch = t_start;
        float avg_batch_ms = -1.0f;
        int batches_per_epoch = ((int)data.size() + batch_size - 1) / batch_size;
        int total_batches = max_epochs > 0 ? max_epochs * batches_per_epoch : 0;
        int batches_done = 0;

        for (int epoch = 0; !interrupt; epoch++) {
            if (max_epochs > 0 && epoch >= max_epochs) break;
            float total_loss = 0;
            int count = 0;

            for (size_t start = 0; start + 1 < data.size(); start += batch_size) {
                int B = std::min((size_t)batch_size, data.size() - 1 - start);
                int pct = (int)((float)start / data.size() * 100);

                if (use_cpu) {
                    train_batch_cpu(data, start, B, total_loss, count);
                    reduce_update(B);
                } else {
                    for (int i = 0; i < B; i++) {
                        ptr_inputs_int[i] = data[start + i];
                        ptr_targets_int[i] = data[start + i + 1];
                    }
                    vk->executeTrainBatch(gpu_inputs, gpu_targets,
                                          gpu_W1, gpu_W2, gpu_b1, gpu_b2,
                                          gpu_hidden_batch, gpu_logits_batch,
                                          gpu_dW1, gpu_dW2, gpu_db1, gpu_db2,
                                          gpu_losses, B, V, H);
                    vk->executeTrainUpdate(gpu_inputs,
                                           gpu_dW1, gpu_dW2, gpu_db1, gpu_db2,
                                           gpu_W1, gpu_W2, gpu_b1, gpu_b2,
                                           B, V, H, lr);
                    for (int i = 0; i < B; i++) {
                        total_loss += ptr_losses[i];
                        count++;
                    }
                    sync_weights_cpu();
                }

                float eta = 0;
                if (max_epochs > 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    float batch_ms = std::chrono::duration<float, std::milli>(now - t_batch).count();
                    t_batch = now;
                    float alpha = 0.005f;
                    if (avg_batch_ms < 0) avg_batch_ms = batch_ms;
                    else avg_batch_ms = (1.0f - alpha) * avg_batch_ms + alpha * batch_ms;
                    batches_done++;
                    int remaining = total_batches - batches_done;
                    eta = avg_batch_ms * remaining / 1000.0f;
                }
                const char* dk_green  = "\033[38;2;0;130;0m";
                const char* bracket  = "\033[38;2;60;140;255m";
                const char* orange   = "\033[38;2;255;165;0m";
                const char* cyan_c   = "\033[38;2;0;200;200m";
                const char* loss_lbl = "\033[38;2;0;160;0m";
                const char* rst      = "\033[0m";
                int filled = pct/5;
                int empty = 20 - filled;
                std::string bar;
                for (int i = 0; i < filled; i++) {
                    float t = (filled > 1) ? (float)i / (filled - 1) : 0.5f;
                    bar += hash_color(t) + "#" + rst;
                }
                bar += std::string(empty, '.');
                float avg_loss = total_loss / count;
                std::string ep_str;
                if (max_epochs > 0)
                    ep_str = epoch_color(epoch, max_epochs) + std::to_string(epoch) + rst
                           + "/" + cyan_c + std::to_string(max_epochs) + rst;
                else
                    ep_str = epoch_color(epoch, max_epochs) + std::to_string(epoch) + rst;
                std::cout << "\r"
                          << star_color(pct / 100.0f) << "\xe2\x9c\xa6" << rst << " "
                          << dk_green << "ep" << rst << " " << ep_str << " "
                          << bracket << "[" << rst << bar << bracket << "]" << rst << " "
                          << loss_lbl << "Loss" << rst << "="
                          << gradient_color(loss_to_t(avg_loss)) << std::fixed << std::setprecision(4) << avg_loss << rst << " "
                          << orange << "eta" << rst << "="
                          << gradient_color(eta_to_t(eta)) << format_eta(eta) << rst
                          << "    " << std::flush;
            }
            std::cout << std::endl;

            if (save_every_epoch && !save_path.empty()) {
                save_model(save_path);
            }

            if (interrupt) {
                std::cout << "\nInterrupted. Saving model...\n" << std::endl;
                if (!save_path.empty()) {
                    save_model(save_path);
                }
                break;
            }
        }
    }

    void set_weights(float* src) {
        memcpy(W1, src, V * H * sizeof(float));
        memcpy(b1, src + V*H, H * sizeof(float));
        memcpy(W2, src + V*H + H, H * V * sizeof(float));
        memcpy(b2, src + V*H + H + H*V, V * sizeof(float));
    }

    void get_weights(float* dst) {
        memcpy(dst, W1, V * H * sizeof(float));
        memcpy(dst + V*H, b1, H * sizeof(float));
        memcpy(dst + V*H + H, W2, H * V * sizeof(float));
        memcpy(dst + V*H + H + H*V, b2, V * sizeof(float));
    }

    static size_t weight_bytes(int V, int H) {
        return (V*H + H + H*V + V) * sizeof(float);
    }

    void save_model(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot save model to " << path << std::endl; return; }
        int v = V, h = H;
        float lr_saved = lr;
        int bs_saved = batch_size;
        float temp_saved = temperature;
        f.write((const char*)&v, sizeof(int));
        f.write((const char*)&h, sizeof(int));
        f.write((const char*)&lr_saved, sizeof(float));
        f.write((const char*)&bs_saved, sizeof(int));
        f.write((const char*)&temp_saved, sizeof(float));
        f.write((const char*)W1, V * H * sizeof(float));
        f.write((const char*)b1, H * sizeof(float));
        f.write((const char*)W2, H * V * sizeof(float));
        f.write((const char*)b2, V * sizeof(float));
        int dlen = (int)backup_dir.size();
        f.write((const char*)&dlen, sizeof(int));
        f.write(backup_dir.data(), dlen);
        std::cout << "Model saved to " << path << std::endl;
    }

    bool load_model(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int v, h;
        float lr_loaded;
        int bs_loaded;
        float temp_loaded;
        f.read((char*)&v, sizeof(int));
        f.read((char*)&h, sizeof(int));
        f.read((char*)&lr_loaded, sizeof(float));
        f.read((char*)&bs_loaded, sizeof(int));
        f.read((char*)&temp_loaded, sizeof(float));
        if (v != V || h != H) {
            std::cerr << "Model file has V=" << v << " H=" << h
                      << " but current is V=" << V << " H=" << H
                      << ". Retraining." << std::endl;
            return false;
        }
        f.read((char*)W1, V * H * sizeof(float));
        f.read((char*)b1, H * sizeof(float));
        f.read((char*)W2, H * V * sizeof(float));
        f.read((char*)b2, V * sizeof(float));
        backup_dir.clear();
        if (f.peek() != EOF) {
            int dlen = 0;
            f.read((char*)&dlen, sizeof(int));
            if (dlen > 0 && dlen < 4096) {
                char* buf = new char[dlen];
                f.read(buf, dlen);
                backup_dir.assign(buf, dlen);
                delete[] buf;
            }
        }
        if (backup_dir.empty()) backup_dir = generate_backup_dir();
        sync_weights_gpu();
        std::cout << "Model loaded from " << path << std::endl;
        lr = lr_loaded;
        batch_size = bs_loaded;
        temperature = temp_loaded;
        return true;
    }

    static bool peek_header(const std::string& path, int& out_V, int& out_H,
                            float& out_lr, int& out_bs, float& out_temp) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.read((char*)&out_V, sizeof(int));
        f.read((char*)&out_H, sizeof(int));
        f.read((char*)&out_lr, sizeof(float));
        f.read((char*)&out_bs, sizeof(int));
        f.read((char*)&out_temp, sizeof(float));
        return true;
    }

    int sample_next(int input_char, std::mt19937& gen, const std::vector<int>& recent) {
        float* logits = new float[V];
        forward_single(input_char, logits);

        float mx = logits[0];
        for (int j = 1; j < V; j++) if (logits[j] > mx) mx = logits[j];

        float* probs = new float[V];
        float se = 0;
        for (int j = 0; j < V; j++) {
            float s = logits[j] / temperature;
            probs[j] = expf(s - mx / temperature);
            se += probs[j];
        }

        int window = 10;
        if ((int)recent.size() >= window) {
            for (int j = 0; j < V; j++) {
                if (std::find(recent.end() - window, recent.end(), j) != recent.end()) {
                    probs[j] /= 2.0f;
                }
            }
        }

        if (top_k > 0 && top_k < V) {
            std::vector<std::pair<float, int>> idx_prob;
            for (int j = 0; j < V; j++) idx_prob.emplace_back(probs[j], j);
            std::partial_sort(idx_prob.begin(), idx_prob.begin() + top_k, idx_prob.end(),
                              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
            float thresh = idx_prob[top_k - 1].first;
            se = 0;
            for (int j = 0; j < V; j++) {
                if (probs[j] < thresh) probs[j] = 0;
                se += probs[j];
            }
        }

        float r = std::uniform_real_distribution<float>(0.0f, se)(gen);
        float cum = 0;
        for (int j = 0; j < V; j++) {
            cum += probs[j];
            if (r < cum) { delete[] logits; delete[] probs; return j; }
        }
        delete[] logits; delete[] probs;
        return V - 1;
    }

    std::string generate(const std::string& prompt, int max_tokens, std::mt19937& gen, Tokenizer& tk) {
        std::string res = prompt;
        std::vector<int> recent;
        int curr = tk.stoi[res.back()];
        recent.push_back(curr);
        for (int i = 0; i < max_tokens; i++) {
            curr = sample_next(curr, gen, recent);
            res += tk.itos[curr];
            recent.push_back(curr);
        }
        return res;
    }

    ~NeuralLM() {
        delete[] W1; delete[] b1; delete[] W2; delete[] b2;
    }
};

static const int MAX_WORKERS = 16;
static const char* bar_colors[] = {
    "31", "32", "33", "34", "35", "36", "91", "92",
    "31", "32", "33", "34", "35", "36", "91", "92"
};

struct SharedProgress {
    std::atomic<int> epoch[MAX_WORKERS];
    std::atomic<float> loss[MAX_WORKERS];
    std::atomic<int> pct[MAX_WORKERS];
};

struct SharedBarrier {
    std::atomic<int> b1_count{0};
    std::atomic<bool> b1_sense{false};
    std::atomic<bool> abort{false};
    std::atomic<bool> coord_ready{false};
    std::atomic<int> coord_ack{0};
};

static void barrier_sense(SharedBarrier* b, int n, bool& local_sense) {
    if (b->abort.load()) return;
    if (b->b1_count.fetch_add(1) == n - 1) {
        b->b1_count.store(0);
        b->b1_sense.store(local_sense);
    } else {
        while (b->b1_sense.load() != local_sense) {
            if (training_interrupted) {
                b->abort.store(true);
                return;
            }
        }
    }
    local_sense = !local_sense;
}

static std::string rgb_fg(int r, int g, int b) {
    return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
}

static float loss_to_t(float loss) {
    return 1.0f / (1.0f + 0.5f * loss);
}

static float eta_to_t(float eta) {
    if (eta < 0 || !std::isfinite(eta)) return 0;
    return 1.0f / (1.0f + 0.1f * eta);
}

static std::string gradient_color(float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    struct RGB { int r, g, b; };
    static const RGB stops[] = {
        {255, 0, 0},
        {235, 72, 242},
        {116, 118, 222},
        {0, 255, 0}
    };
    float segment = t * 3.0f;
    int idx = (int)segment;
    if (idx >= 3) idx = 2;
    float frac = segment - idx;
    int r = (int)(stops[idx].r * (1 - frac) + stops[idx + 1].r * frac + 0.5f);
    int g = (int)(stops[idx].g * (1 - frac) + stops[idx + 1].g * frac + 0.5f);
    int b = (int)(stops[idx].b * (1 - frac) + stops[idx + 1].b * frac + 0.5f);
    return rgb_fg(r, g, b);
}

static std::string star_color(float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    int r = (int)(255 * (1 - t) + 0.5f);
    int g = (int)(255 * t + 0.5f);
    int b = (int)(255 * t + 0.5f);
    return rgb_fg(r, g, b);
}

static std::string hash_color(float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    int r = (int)(116 + t * (235 - 116) + 0.5f);
    int g = (int)(118 + t * (72 - 118) + 0.5f);
    int b = (int)(222 + t * (242 - 222) + 0.5f);
    return rgb_fg(r, g, b);
}

static std::string epoch_color(int epoch, int max_epochs) {
    if (max_epochs <= 0) return rgb_fg(116, 118, 222);
    float t = std::max(0.0f, std::min(1.0f, (float)epoch / max_epochs));
    int r = (int)(255 * (1 - t) + 0.5f);
    int g = (int)(200 * t + 0.5f);
    int b = (int)(200 * t + 0.5f);
    return rgb_fg(r, g, b);
}

static std::string format_eta(float secs) {
    if (secs < 0 || !std::isfinite(secs)) return "--";
    int total = (int)secs;
    int h = total / 3600;
    int m = (total % 3600) / 60;
    int s = total % 60;
    if (h > 0) return std::to_string(h) + "h" + std::to_string(m) + "m";
    if (m > 0) return std::to_string(m) + "m" + std::to_string(s) + "s";
    return std::to_string(s) + "s";
}

static void render_lines(int n, int* epochs, float* losses, int* pcts, bool winner, float eta_secs = 0, bool = false, int max_epochs = 0) {
    std::cout << "\033[u";
    const char* dk_green  = "\033[38;2;0;130;0m";
    const char* bracket   = "\033[38;2;60;140;255m";
    const char* orange    = "\033[38;2;255;165;0m";
    const char* cyan_c    = "\033[38;2;0;200;200m";
    const char* loss_lbl  = "\033[38;2;0;160;0m";
    const char* rst       = "\033[0m";
    for (int w = 0; w < n; w++) {
        int p = std::min(pcts[w], 100);
        int filled = p / 5;
        int empty = 20 - filled;
        std::string bar;
        for (int i = 0; i < filled; i++) {
            float t = (filled > 1) ? (float)i / (filled - 1) : 0.5f;
            bar += hash_color(t) + "#" + rst;
        }
        bar += std::string(empty, '.');
        std::string ep_str;
        if (max_epochs > 0)
            ep_str = epoch_color(epochs[w], max_epochs) + std::to_string(epochs[w]) + rst
                   + "/" + cyan_c + std::to_string(max_epochs) + rst;
        else
            ep_str = epoch_color(epochs[w], max_epochs) + std::to_string(epochs[w]) + rst;
        std::cout << "\033[2K\033[" << bar_colors[w % 16] << "m[w" << w << "]"
                  << (winner && w == 0 ? " \xe2\x98\x85" : "  ")
                  << "\033[0m "
                  << star_color(p / 100.0f) << "\xe2\x9c\xa6" << rst << " "
                  << dk_green << "ep" << rst << " " << ep_str << " "
                  << bracket << "[" << rst << bar << bracket << "]" << rst << " "
                  << loss_lbl << "Loss" << rst << "="
                  << gradient_color(loss_to_t(losses[w])) << std::fixed << std::setprecision(4) << losses[w] << rst << " "
                  << orange << "eta" << rst << "="
                  << gradient_color(eta_to_t(eta_secs)) << format_eta(eta_secs) << rst
                  << "\n";
    }
    std::cout << std::flush;
}

static void train_parallel_fork(const std::vector<int>& data, int V, int H,
                                 float lr, int batch_size, float temperature,
                                 int num_cores, int num_workers, const std::string& save_path,
                                 int max_epochs = 0, bool continue_training = false,
                                 int top_k = 20, bool save_every_epoch = false,
                                 bool use_cpu = false) {
    size_t wb = NeuralLM::weight_bytes(V, H);
    size_t shm_size = sizeof(SharedBarrier) + sizeof(SharedProgress) + wb * num_workers;
    void* shm = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    SharedBarrier* barrier = (SharedBarrier*)shm;
    SharedProgress* prog = (SharedProgress*)((char*)shm + sizeof(SharedBarrier));
    float* weight_slots = (float*)((char*)shm + sizeof(SharedBarrier) + sizeof(SharedProgress));

    if (continue_training) {
        std::ifstream f(save_path, std::ios::binary);
        if (f) {
            int v, h, bs_s;
            float lr_s, temp_s;
            f.read((char*)&v, sizeof(int));
            f.read((char*)&h, sizeof(int));
            f.read((char*)&lr_s, sizeof(float));
            f.read((char*)&bs_s, sizeof(int));
            f.read((char*)&temp_s, sizeof(float));
            if (v != V || h != H) {
                std::cerr << "Model file has V=" << v << " H=" << h
                          << " but current is V=" << V << " H=" << H
                          << ". Starting from scratch.\n";
            } else {
                f.read((char*)weight_slots, wb);
                lr = lr_s;
                batch_size = bs_s;
                std::cout << "Continuing parallel training from " << save_path << "\n" << std::endl;
            }
        } else {
            std::cout << "No saved model to continue from. Starting from scratch.\n" << std::endl;
        }
    }

    pid_t children[num_workers];
    for (int w = 0; w < num_workers; w++) {
        pid_t pid = fork();
        if (pid == 0) {
            int slice_start = (size_t)w * data.size() / num_workers;
            int slice_end = (size_t)(w + 1) * data.size() / num_workers;
            if (slice_end > (int)data.size()) slice_end = data.size();
            if (slice_end - slice_start < 2) exit(0);

            std::vector<int> slice(data.begin() + slice_start, data.begin() + slice_end);
            float* my_slot = weight_slots + w * (wb / sizeof(float));
            memcpy(my_slot, weight_slots, wb);

            VulkanCompute vk_child;
            NeuralLM model(vk_child, V, H, lr, batch_size, temperature, top_k, use_cpu);
            model.set_weights(my_slot);

            bool local_sense = true;
            auto t_start = std::chrono::high_resolution_clock::now();
            int batches_per_epoch = ((int)slice.size() + batch_size - 1) / batch_size;
            int total_batches = max_epochs > 0 ? max_epochs * batches_per_epoch : 0;
            int batches_done = 0;

            if (w == 0) {
                std::cout << "\033[s";
                for (int i = 0; i < num_workers; i++) std::cout << "\n";
                std::cout << "\033[u" << std::flush;
            }

            for (int epoch = 0; !training_interrupted; epoch++) {
                if (max_epochs > 0 && epoch >= max_epochs) break;
                if (barrier->abort.load()) break;
                float total_loss = 0;
                int count = 0;
                int batches = 0;
                for (size_t start = 0; start + 1 < slice.size(); start += batch_size) {
                    int B = std::min((size_t)batch_size, slice.size() - 1 - start);
                    for (int i = 0; i < B; i++) {
                        model.ptr_inputs_int[i] = slice[start + i];
                        model.ptr_targets_int[i] = slice[start + i + 1];
                    }
                    model.vk->executeTrainBatch(
                        model.gpu_inputs, model.gpu_targets,
                        model.gpu_W1, model.gpu_W2, model.gpu_b1, model.gpu_b2,
                        model.gpu_hidden_batch, model.gpu_logits_batch,
                        model.gpu_dW1, model.gpu_dW2, model.gpu_db1, model.gpu_db2,
                        model.gpu_losses, B, V, H);

                    model.vk->executeTrainUpdate(
                        model.gpu_inputs,
                        model.gpu_dW1, model.gpu_dW2, model.gpu_db1, model.gpu_db2,
                        model.gpu_W1, model.gpu_W2, model.gpu_b1, model.gpu_b2,
                        B, V, H, lr);

                    for (int i = 0; i < B; i++) { total_loss += model.ptr_losses[i]; count++; }
                    model.sync_weights_cpu();

                    prog->epoch[w].store(epoch, std::memory_order_release);
                    prog->loss[w].store(total_loss / count, std::memory_order_release);
                    prog->pct[w].store((int)((float)start / slice.size() * 100), std::memory_order_release);

                    if (w == 0) {
                        batches_done++;
                        if (batches % 5 == 0) {
                            int e_[MAX_WORKERS], p_[MAX_WORKERS]; float l_[MAX_WORKERS];
                            for (int ww = 0; ww < num_workers; ww++) {
                                e_[ww] = prog->epoch[ww].load(std::memory_order_acquire);
                                l_[ww] = prog->loss[ww].load(std::memory_order_acquire);
                                p_[ww] = prog->pct[ww].load(std::memory_order_acquire);
                            }
                            float eta = 0;
                            if (max_epochs > 0 && batches_done > 0) {
                                auto now = std::chrono::high_resolution_clock::now();
                                float elapsed = std::chrono::duration<float>(now - t_start).count();
                                float progress = (float)batches_done / total_batches;
                                if (progress > 0) eta = elapsed / progress - elapsed;
                            }
                            render_lines(num_workers, e_, l_, p_, false, eta, save_every_epoch, max_epochs);
                        }
                    }
                    batches++;
                }

                float avg_loss = total_loss / count;
                model.get_weights(my_slot);
                prog->epoch[w].store(epoch, std::memory_order_release);
                prog->loss[w].store(avg_loss, std::memory_order_release);
                prog->pct[w].store(100, std::memory_order_release);

                barrier_sense(barrier, num_workers, local_sense);

                if (w == 0) {
                    int best_w = 0;
                    float best_loss = prog->loss[0].load(std::memory_order_acquire);
                    for (int ww = 1; ww < num_workers; ww++) {
                        float l = prog->loss[ww].load(std::memory_order_acquire);
                        if (l < best_loss) { best_loss = l; best_w = ww; }
                    }
                    float* best_slot = weight_slots + best_w * (wb / sizeof(float));
                    float* slot0 = weight_slots;
                    if (best_w != 0) memcpy(slot0, best_slot, wb);
                    for (int ww = 0; ww < num_workers; ww++) {
                        float* ws = weight_slots + ww * (wb / sizeof(float));
                        if (ws != slot0) memcpy(ws, slot0, wb);
                    }

                    int e_[MAX_WORKERS], p_[MAX_WORKERS]; float l_[MAX_WORKERS];
                    for (int ww = 0; ww < num_workers; ww++) {
                        e_[ww] = prog->epoch[ww].load(std::memory_order_acquire);
                        l_[ww] = prog->loss[ww].load(std::memory_order_acquire);
                        p_[ww] = prog->pct[ww].load(std::memory_order_acquire);
                    }
                    float eta = 0;
                    if (max_epochs > 0 && batches_done > 0) {
                        auto now = std::chrono::high_resolution_clock::now();
                        float elapsed = std::chrono::duration<float>(now - t_start).count();
                        float progress = (float)batches_done / total_batches;
                        if (progress > 0) eta = elapsed / progress - elapsed;
                    }
                    render_lines(num_workers, e_, l_, p_, true, eta, save_every_epoch, max_epochs);
                    barrier->coord_ready.store(true, std::memory_order_release);
                } else {
                    while (!barrier->coord_ready.load(std::memory_order_acquire)) {
                        if (barrier->abort.load()) break;
                    }
                    barrier->coord_ack.fetch_add(1, std::memory_order_acq_rel);
                }

                if (w == 0) {
                    while (barrier->coord_ack.load(std::memory_order_acquire) < num_workers - 1) {
                        if (barrier->abort.load()) break;
                    }
                    barrier->coord_ready.store(false, std::memory_order_release);
                    barrier->coord_ack.store(0, std::memory_order_release);
                }

                memcpy(my_slot, weight_slots, wb);
                model.set_weights(my_slot);

                if (save_every_epoch && w == 0) {
                    model.save_model(save_path);
                }

                if (training_interrupted || barrier->abort.load()) {
                    if (w == 0) {
                        model.save_model(save_path);
                        std::cout << "\n\033[32m✓ Interrupted. Model saved (best weights).\033[0m\n";
                    }
                    break;
                }
            }

            model.destroy_buffers(vk_child);
            exit(0);
        } else {
            children[w] = pid;
        }
    }

    for (int w = 0; w < num_workers; w++) waitpid(children[w], nullptr, 0);
    std::cout << "\033[" << num_workers << "B" << std::flush;

    std::string parent_bdir = read_backup_dir_from_file(save_path);
    if (parent_bdir.empty()) parent_bdir = NeuralLM::generate_backup_dir();
    backup_model(save_path);
    std::ofstream f(save_path, std::ios::binary);
    if (f) {
        int v = V, h = H;
        f.write((const char*)&v, sizeof(int));
        f.write((const char*)&h, sizeof(int));
        float lr_s = lr; f.write((const char*)&lr_s, sizeof(float));
        int bs_s = batch_size; f.write((const char*)&bs_s, sizeof(int));
        float t_s = temperature; f.write((const char*)&t_s, sizeof(float));
        f.write((const char*)weight_slots, wb);
        int dlen = (int)parent_bdir.size();
        f.write((const char*)&dlen, sizeof(int));
        f.write(parent_bdir.data(), dlen);
        std::cout << "Model saved to " << save_path << std::endl;
    }
    munmap(shm, shm_size);
}

struct ChatSession {
    NeuralLM& model;
    Tokenizer& tk;
    int max_tokens;
    std::mt19937 gen;
    std::thread worker;
    std::queue<std::string> input_queue;
    std::queue<std::string> output_queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running{true};

    ChatSession(NeuralLM& m, Tokenizer& t, int _max_tokens)
        : model(m), tk(t), max_tokens(_max_tokens), gen(42) {
        worker = std::thread(&ChatSession::process_loop, this);
    }

    void send_message(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mtx);
        input_queue.push(msg);
        cv.notify_one();
    }

    bool poll_response(std::string& out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (output_queue.empty()) return false;
        out = output_queue.front();
        output_queue.pop();
        return true;
    }

    void process_loop() {
        while (running) {
            std::string msg;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return !input_queue.empty() || !running; });
                if (!running) return;
                msg = input_queue.front();
                input_queue.pop();
            }
            std::string clean;
            for (char c : msg) if (tk.stoi.count(c)) clean += c;
            if (clean.empty()) clean = "T";
            std::string resp = model.generate(clean, max_tokens, gen, tk);
            std::lock_guard<std::mutex> lock(mtx);
            output_queue.push(resp);
        }
    }

    ~ChatSession() {
        running = false;
        cv.notify_one();
        if (worker.joinable()) worker.join();
    }
};

std::string load_corpus(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool has_suffix(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.substr(s.size() - suf.size()) == suf;
}

static void scan_trdat(const std::string& dirpath, std::vector<std::string>& files) {
    DIR* dir = opendir(dirpath.c_str());
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dirpath + "/" + name;
        if (entry->d_type == DT_DIR) {
            scan_trdat(full, files);
        } else if (has_suffix(name, ".trdat") || has_suffix(name, ".trdata")) {
            files.push_back(full);
        }
    }
    closedir(dir);
}

static bool is_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return true;
    char buf[4096] = {0};
    f.read(buf, sizeof(buf));
    std::streamsize n = f.gcount();
    for (std::streamsize i = 0; i < n; i++) {
        if (buf[i] == '\0') return true;
    }
    return false;
}

static void scan_all_files(const std::string& dirpath, std::vector<std::string>& files) {
    DIR* dir = opendir(dirpath.c_str());
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dirpath + "/" + name;
        if (entry->d_type == DT_DIR) {
            if (name[0] != '.' && name != "bin" && name != "lib" && name != "node_modules" && name != ".git") {
                scan_all_files(full, files);
            }
        } else if (entry->d_type == DT_REG && !is_binary(full)) {
            files.push_back(full);
        }
    }
    closedir(dir);
}

static std::string collect_corpus(const std::string& datadir) {
    std::vector<std::string> trdat_files;

    scan_trdat(datadir, trdat_files);
    if (datadir != "." && datadir != "TrainingData") {
        struct stat st;
        if (stat("TrainingData", &st) == 0 && S_ISDIR(st.st_mode)) {
            std::cout << "Also scanning TrainingData/ for .trdat/.trdata files...\n";
            scan_trdat("TrainingData", trdat_files);
        }
    }

    if (!trdat_files.empty()) {
        std::sort(trdat_files.begin(), trdat_files.end());
        trdat_files.erase(std::unique(trdat_files.begin(), trdat_files.end()), trdat_files.end());
        std::string result;
        for (const auto& f : trdat_files) {
            std::string content = load_corpus(f);
            if (!content.empty()) {
                std::cout << "\x1b[32mLoaded corpus:\x1b[0m \x1b[38;5;21m" << f << " \x1b[31m(" << content.size() << " chars)\x1b[0m" << std::endl;
                result += content;
                result += '\n';
            }
        }
        if (!result.empty()) return result;
    }

    std::string cp = (datadir == ".") ? "corpus.txt" : datadir + "/corpus.txt";
    std::string fallback = load_corpus(cp);
    if (!fallback.empty()) {
        std::cout << "No .trdat/.trdata files found, loaded " << cp << " (" << fallback.size() << " chars)" << std::endl;
        return fallback;
    }

    const char* home = getenv("HOME");
    if (home) {
        std::cout << "No training files found. Scanning ~ for non-binary files...\n";
        std::vector<std::string> all_files;
        scan_all_files(home, all_files);
        if (!all_files.empty()) {
            std::string result;
            int loaded = 0;
            for (const auto& f : all_files) {
                std::string content = load_corpus(f);
                if (!content.empty()) {
                    bool has_text = false;
                    for (char c : content) {
                        if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\v' && c != '\f') {
                            has_text = true;
                            break;
                        }
                    }
                    if (!has_text) continue;
                    result += content;
                    result += '\n';
                    loaded++;
                    if (result.size() > 5000000) break;
                }
            }
            if (!result.empty()) {
                std::cout << "Loaded " << loaded << " files from ~ (" << result.size() << " chars)" << std::endl;
                return result;
            }
        }
    }

    std::cerr << "Error: no training data found anywhere." << std::endl;
    exit(1);
}

static void run_chat(NeuralLM& model, Tokenizer& tk, int max_tokens) {
    std::cout << "\033[1;33m=== CHAT MODE ===\033[0m\n";
    std::cout << "Type a prompt and press Enter. Model continues from your text.\n";
    std::cout << "Max tokens: " << max_tokens << " (change with --max-tokens)\n";
    std::cout << "Type 'exit' to quit.\n" << std::endl;

    ChatSession chat(model, tk, max_tokens);

    auto gen_and_show = [&](const std::string& p, bool show_prompt) {
        auto tg0 = std::chrono::high_resolution_clock::now();
        chat.send_message(p);
        std::string full;
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (chat.poll_response(full)) break;
        }
        auto tg1 = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(tg1 - tg0).count();
        std::string g = full.size() > p.size() ? full.substr(p.size()) : full;
        if (show_prompt) std::cout << "\033[1;34m" << p << "\033[0m";
        std::cout << g << "\n  \033[2m(" << dt << "s, " << (g.empty() ? 0 : (int)(g.size()/dt)) << " cps)\033[0m" << std::endl;
    };

    gen_and_show("To be", true);

    while (true) {
        std::cout << "\n\033[1;32m> \033[0m" << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (line == "exit" || line == "quit") break;
        if (!line.empty()) gen_and_show(line, true);
    }
}

static std::string read_backup_dir_from_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    int V, H;
    f.read((char*)&V, sizeof(int));
    f.read((char*)&H, sizeof(int));
    f.seekg(3 * sizeof(float), std::ios::cur);
    std::streampos weights_end = 20 + (std::streampos)(
        V * sizeof(float) * H + H * sizeof(float) +
        H * sizeof(float) * V + V * sizeof(float));
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    if (file_size <= weights_end + (std::streamoff)4) return "";
    f.seekg(weights_end);
    int dlen = 0;
    f.read((char*)&dlen, sizeof(int));
    if (dlen <= 0 || dlen > 4096 || weights_end + (std::streamoff)(4 + dlen) > file_size) return "";
    char* buf = new char[dlen];
    f.read(buf, dlen);
    std::string result(buf, dlen);
    delete[] buf;
    return result;
}

static void backup_model(const std::string& path) {
    std::ifstream src(path, std::ios::binary);
    if (!src) return;
    std::string dir = read_backup_dir_from_file(path);
    if (dir.empty()) {
        size_t slash = path.rfind('/');
        std::string d = (slash == std::string::npos) ? "." : path.substr(0, slash);
        dir = d + "/ModelBackups";
    }
    mkdir(dir.c_str(), 0755);
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    size_t slash = path.rfind('/');
    std::string fname = (slash == std::string::npos) ? path : path.substr(slash + 1);
    std::string backup = dir + "/" + fname + "." + std::to_string(t);
    std::ofstream dst(backup, std::ios::binary);
    if (dst) {
        dst << src.rdbuf();
        std::cout << "Backed up to " << backup << std::endl;
    }
}

int main(int argc, char* argv[]) {
    bool force_train = false;
    int num_cores = std::max(1u, std::thread::hardware_concurrency());
    int n_embd = 256;
    float lr = 0.06f;
    int batch_size = 256;
    float temperature = 0.7f;
    int top_k = 20;
    size_t max_train_chars = 1000000;
    int max_gen_tokens = 500;
    std::string load_path = "model.bin";
    std::string save_path = "model.bin";
    bool save_every_epoch = false;
    int parallel_workers = 1;
    int max_epochs = 0;
    bool show_model_info = false;
    std::string data_dir = ".";
    bool continue_training = false;
    bool want_chat = false;

    auto print_help = [&]() {
        std::cout << "Usage: random_llm [OPTIONS]\n"
                  << "Vulkan-accelerated character-level neural language model.\n"
                  << "First run auto-trains until Ctrl+C, then chats. Subsequent runs load saved model.\n"
                  << "\nOptions:\n"
                  << "  -t, --train          Force retrain from scratch\n"
                  << "  -c, --core N         CPU threads for gradient reduction (default: max)\n"
                  << "  -e, --embed N        Embedding/hidden size (default: 256)\n"
                  << "  -l, --lr F           Learning rate (default: 0.06)\n"
                  << "  -b, --batch N        Mini-batch size (default: 256)\n"
                  << "  -T, --temp F         Sampling temperature (default: 0.7)\n"
                  << "  -k, --topk N         Top-K sampling (default: 20, 0=disabled)\n"
                  << "  -M, --max-train N    Max training characters (default: 1000000)\n"
                  << "  -n, --max-tokens N   Max generation tokens per reply (default: 500)\n"
                  << "  -m, --model PATH     Set both load and save model path (default: model.bin)\n"
                  << "      --LoadModel PATH Model file path to load from\n"
                  << "      --SaveModel PATH Model file path to save to\n"
                  << "      --DataDir PATH   Training data directory with .trdat/.trdata (default: .)\n"
                  << "                     Also checks TrainingData/ alongside. Falls back to\n"
                  << "                     corpus.txt, then scans ~/ for non-binary files.\n"
                  << "      --SaveEpoch      Save model after every epoch (no backup)\n"
                  << "  -p, --parallel N     Fork N workers, each trains on data slice, sync weights (default: 1)\n"
                  << "  -I, --iter N         Train for N epochs, then stop (default: infinite)\n"
                  << "      --continue       Continue training from saved model (overrides --train)\n"
                  << "      --chat           Force chat mode after training\n"
                  << "      --ModelInfo      Show saved model info and exit\n"
                  << "  -h, --help           Show this help and exit\n";
    };

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_help();
            return 0;
        } else if (a == "-t" || a == "--train") {
            force_train = true;
        } else if ((a == "-c" || a == "--core") && i + 1 < argc) {
            num_cores = std::atoi(argv[++i]);
            if (num_cores < 1) num_cores = 1;
            int max_cores = (int)std::thread::hardware_concurrency();
            if (num_cores > max_cores) {
                std::cout << "Capping cores to available " << max_cores << std::endl;
                num_cores = max_cores;
            }
        } else if ((a == "-e" || a == "--embed") && i + 1 < argc) {
            n_embd = std::atoi(argv[++i]);
            if (n_embd < 1) n_embd = 1;
        } else if ((a == "-l" || a == "--lr") && i + 1 < argc) {
            lr = std::atof(argv[++i]);
        } else if ((a == "-b" || a == "--batch") && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
            if (batch_size < 1) batch_size = 1;
        } else if ((a == "-T" || a == "--temp") && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if ((a == "-k" || a == "--topk") && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
            if (top_k < 0) top_k = 0;
        } else if ((a == "-M" || a == "--max-train") && i + 1 < argc) {
            max_train_chars = std::atoi(argv[++i]);
        } else if ((a == "-n" || a == "--max-tokens") && i + 1 < argc) {
            max_gen_tokens = std::atoi(argv[++i]);
            if (max_gen_tokens < 1) max_gen_tokens = 1;
        } else if ((a == "-m" || a == "--model") && i + 1 < argc) {
            load_path = save_path = argv[++i];
        } else if ((a == "--LoadModel") && i + 1 < argc) {
            load_path = argv[++i];
        } else if ((a == "--SaveModel") && i + 1 < argc) {
            save_path = argv[++i];
        } else if ((a == "--DataDir") && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (a == "--SaveEpoch") {
            save_every_epoch = true;
        } else if ((a == "-p" || a == "--parallel") && i + 1 < argc) {
            parallel_workers = std::atoi(argv[++i]);
            if (parallel_workers < 1) parallel_workers = 1;
        } else if ((a == "-I" || a == "--iter") && i + 1 < argc) {
            max_epochs = std::atoi(argv[++i]);
            if (max_epochs < 1) max_epochs = 1;
        } else if (a == "--continue") {
            continue_training = true;
        } else if (a == "--chat") {
            want_chat = true;
        } else if ((a == "-H" || a == "--max-chars") && i + 1 < argc) {
            max_train_chars = std::atoi(argv[++i]);
        } else if (a == "--ModelInfo") {
            show_model_info = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_help();
            return 1;
        }
    }

    if (show_model_info) {
        std::ifstream f(load_path, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open model file: " << load_path << std::endl;
            return 1;
        }
        int v, h, bs_s;
        float lr_s, temp_s;
        f.read((char*)&v, sizeof(int));
        f.read((char*)&h, sizeof(int));
        f.read((char*)&lr_s, sizeof(float));
        f.read((char*)&bs_s, sizeof(int));
        f.read((char*)&temp_s, sizeof(float));
        std::cout << "Model file: " << load_path << "\n"
                  << "  Vocab size (V):    " << v << "\n"
                  << "  Hidden size (H):   " << h << "\n"
                  << "  Parameters:        " << (v * h * 2 + h + v) << "\n"
                  << "  Learning rate:     " << lr_s << "\n"
                  << "  Batch size:        " << bs_s << "\n"
                  << "  Temperature:       " << temp_s << "\n";
        return 0;
    }

    bool force_cpu = false;
    {
        double ram_mb = check_ram();
        if (ram_mb < 200.0) {
            std::cout << "Low memory (" << (int)ram_mb << " MB available). ";
            std::string cmd = "pgrep -c random_llm 2>/dev/null";
            FILE* fp = popen(cmd.c_str(), "r");
            int count = 0;
            if (fp) { char buf[64]; if (fgets(buf, sizeof(buf), fp)) count = std::atoi(buf); pclose(fp); }
            if (count <= 1) {
                std::cout << "Waiting up to 50s for memory to free...\n" << std::flush;
                for (int i = 0; i < 50; i++) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    ram_mb = check_ram();
                    if (ram_mb >= 200.0) break;
                }
                ram_mb = check_ram();
            }
            if (ram_mb < 200.0) {
                std::cout << "Still low. Falling back to CPU-only mode.\n";
                force_cpu = true;
            }
        }
    }

    try {
        std::string corpus = collect_corpus(data_dir);
        if (corpus.size() > max_train_chars) {
            std::cout << "\x1b[38;5;241mTruncating to " << max_train_chars << " chars for training\x1b[0m\n";
            corpus.resize(max_train_chars);
        }

        Tokenizer tk;
        tk.build(corpus);

        std::vector<int> data;
        for (char c : corpus) data.push_back(tk.stoi[c]);

        {
            int peek_V, peek_H;
            float peek_lr, peek_temp;
            int peek_bs;
            if (NeuralLM::peek_header(load_path, peek_V, peek_H, peek_lr, peek_bs, peek_temp)) {
                if (peek_V == tk.vocab_size) {
                    if (force_train) {
                        bool diff = (peek_H != n_embd) ||
                                    (std::abs(peek_lr - lr) > 1e-6f) ||
                                    (peek_bs != batch_size) ||
                                    (std::abs(peek_temp - temperature) > 1e-6f);
                        if (diff) {
                            std::cout << "Hyperparameters changed (" << load_path
                                      << " has H=" << peek_H << " lr=" << peek_lr
                                      << " batch=" << peek_bs << " temp=" << peek_temp
                                      << "). Backing up old model...\n";
                            backup_model(save_path);
                        }
                    } else {
                        if (peek_H != n_embd) {
                            std::cout << "Auto-detected H=" << peek_H << " from " << load_path
                                      << " (overriding default H=" << n_embd << ")\n";
                            n_embd = peek_H;
                        }
                        lr = peek_lr;
                        batch_size = peek_bs;
                        temperature = peek_temp;
                    }
                }
            }
        }

        std::cout << "\033[1;36m╔══════════════════════════════════════╗\033[0m\n";
        std::cout << "\033[1;36m║     RANDOM NEURAL LANGUAGE MODEL    ║\033[0m\n";
        std::cout << "\033[1;36m╚══════════════════════════════════════╝\033[0m\n";
        std::cout << "Vocab: " << tk.vocab_size << " chars, "
                  << "Params: " << (tk.vocab_size * n_embd * 2 + n_embd + tk.vocab_size) << "\n";
        if (force_cpu)
            std::cout << "Mode: CPU-only";
        else
            std::cout << "Mode: GPU" /* +Vulkan" /* + CPU (" << num_cores << " cores)"*/;
        std::cout << "\n" << std::endl;

        if (!force_cpu && parallel_workers > 1) {
            std::cout << "Parallel workers: " << parallel_workers << "\n" << std::endl;
            backup_model(save_path);
            std::signal(SIGINT, handle_sigint);
            auto t0 = std::chrono::high_resolution_clock::now();
            train_parallel_fork(data, tk.vocab_size, n_embd, lr, batch_size, temperature,
                                num_cores, parallel_workers, save_path, max_epochs, continue_training, top_k, save_every_epoch, force_cpu);
            auto t1 = std::chrono::high_resolution_clock::now();
            float secs = std::chrono::duration<float>(t1 - t0).count();
            std::cout << "\n\033[1;32m✓ Parallel training done in " << std::fixed
                      << std::setprecision(1) << secs << "s\033[0m" << std::endl;

            bool any_training = force_train || continue_training;
            if (want_chat || !any_training) {
                std::cout << "Loading model for chat..." << std::endl;
                std::signal(SIGINT, SIG_DFL);
                VulkanCompute vk;
                NeuralLM model(vk, tk.vocab_size, n_embd, lr, batch_size, temperature, top_k, force_cpu);
                model.load_model(load_path);
                std::cout << std::endl;
                run_chat(model, tk, max_gen_tokens);
            }
        } else {
            VulkanCompute vk;
            NeuralLM model(vk, tk.vocab_size, n_embd, lr, batch_size, temperature, top_k, force_cpu);

            bool should_train = false;
            if (continue_training) {
                if (model.load_model(load_path)) {
                    std::cout << "Continuing training from saved model (" << load_path << ").\n" << std::endl;
                } else {
                    std::cout << "No saved model to continue from. Training from scratch.\n" << std::endl;
                }
                should_train = true;
            } else if (force_train) {
                std::cout << "Force re-train requested (--train).\n" << std::endl;
                should_train = true;
            } else if (!model.load_model(load_path)) {
                std::cout << "No saved model found. Training from scratch.\n" << std::endl;
                should_train = true;
            }

            if (should_train) {
                std::signal(SIGINT, handle_sigint);

                auto t0 = std::chrono::high_resolution_clock::now();
                model.train_hybrid(data, num_cores, training_interrupted, max_epochs, save_path, save_every_epoch);
                auto t1 = std::chrono::high_resolution_clock::now();
                float secs = std::chrono::duration<float>(t1 - t0).count();

                backup_model(save_path);
                model.save_model(save_path);

                std::cout << "\033[1;32m✓ Trained in " << std::fixed << std::setprecision(1) << secs << "s\033[0m\n" << std::endl;
                std::signal(SIGINT, SIG_DFL);
            }

            if (want_chat || !should_train) {
                run_chat(model, tk, max_gen_tokens);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\x1b[31mError: \x1b[0m" << e.what() << std::endl;
        return 1;
    }
    return 0;
}
