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
#include <cuda_runtime.h>

#define CUDA_CHECK(val) { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

static std::atomic<bool> training_interrupted{false};
extern "C" void handle_sigint(int) {
    training_interrupted = true;
}

static void check_ram() {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    double avail_mb = (double)pages * page_size / (1024.0 * 1024.0);
    if (avail_mb < 200.0) {
        std::cerr << "Error: Low memory (" << (int)avail_mb << " MB available). "
                  << "Need at least 200 MB. Exiting.\n";
        exit(1);
    }
}

static std::string format_eta(float secs);

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

// === CUDA Kernels ===

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void train_batch_kernel(
    const int* input_indices, const int* target_indices,
    const float* W1, const float* W2, const float* b1, const float* b2,
    float* hidden, float* logits,
    float* dW1_per, float* dW2_per, float* db1_per, float* db2_per,
    float* losses, int B, int V, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    int ci = input_indices[idx];
    int target = target_indices[idx];

    // 1. hidden = ReLU(W1[ci,:] + b1)
    for (int k = 0; k < H; k++) {
        float h = W1[ci * H + k] + b1[k];
        hidden[idx * H + k] = (h > 0.0f) ? h : 0.0f;
    }

    // 2. logits = hidden @ W2 + b2
    for (int j = 0; j < V; j++) {
        float s = b2[j];
        for (int k = 0; k < H; k++) {
            s += hidden[idx * H + k] * W2[k * V + j];
        }
        logits[idx * V + j] = s;
    }

    // 3. softmax (overwrite logits buffer with probs)
    float mx = logits[idx * V];
    for (int j = 1; j < V; j++) {
        if (logits[idx * V + j] > mx) mx = logits[idx * V + j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < V; j++) {
        float e = expf(logits[idx * V + j] - mx);
        logits[idx * V + j] = e;
        sum_exp += e;
    }
    for (int j = 0; j < V; j++) {
        logits[idx * V + j] /= sum_exp;
    }

    // 4. loss
    float p_target = logits[idx * V + target];
    losses[idx] = -logf(p_target > 1e-10f ? p_target : 1e-10f);

    // 5. backward: dW2, dW1, db1
    for (int k = 0; k < H; k++) {
        float hk = hidden[idx * H + k];
        float dh = 0.0f;
        for (int j = 0; j < V; j++) {
            float p = logits[idx * V + j];
            float d = p - (j == target ? 1.0f : 0.0f);
            dW2_per[idx * H * V + k * V + j] = hk * d;
            dh += d * W2[k * V + j];
        }
        float orig = W1[ci * H + k] + b1[k];
        dh = (orig > 0.0f) ? dh : 0.0f;
        dW1_per[idx * H + k] = dh;
        db1_per[idx * H + k] = dh;
    }

    // 6. backward: db2
    for (int j = 0; j < V; j++) {
        float p = logits[idx * V + j];
        float d = p - (j == target ? 1.0f : 0.0f);
        db2_per[idx * V + j] = d;
    }
}

__global__ void train_update_kernel(
    const int* input_indices,
    const float* dW1_per, const float* dW2_per, const float* db1_per, const float* db2_per,
    float* W1, float* W2, float* b1, float* b2,
    int B, int V, int H, float lr, float invB
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int w1_size = V * H;
    int w2_size = H * V;
    int b1_size = H;
    int b2_size = V;
    int total_params = w1_size + w2_size + b1_size + b2_size;

    if (id >= total_params) return;

    if (id < w1_size) {
        int r = id / H;
        int c = id % H;
        float sum = 0.0f;
        for (int i = 0; i < B; i++) {
            if (input_indices[i] == r) {
                sum += dW1_per[i * H + c];
            }
        }
        W1[r * H + c] -= lr * sum * invB;
    } else if (id < w1_size + w2_size) {
        int id2 = id - w1_size;
        int r = id2 / V;
        int c = id2 % V;
        float sum = 0.0f;
        for (int i = 0; i < B; i++) {
            sum += dW2_per[i * H * V + r * V + c];
        }
        W2[r * V + c] -= lr * sum * invB;
    } else if (id < w1_size + w2_size + b1_size) {
        int id2 = id - w1_size - w2_size;
        float sum = 0.0f;
        for (int i = 0; i < B; i++) {
            sum += db1_per[i * H + id2];
        }
        b1[id2] -= lr * sum * invB;
    } else {
        int id2 = id - w1_size - w2_size - b1_size;
        float sum = 0.0f;
        for (int i = 0; i < B; i++) {
            sum += db2_per[i * V + id2];
        }
        b2[id2] -= lr * sum * invB;
    }
}

// === Neural LM ===

struct NeuralLM {
    int V, H;
    float lr;
    int batch_size;
    float temperature;
    int top_k;

    // Weights (Unified Memory)
    float *W1, *b1, *W2, *b2;

    // Single-char inference buffers (Unified Memory)
    float *in, *hidden, *logits;

    // Batch training buffers (Unified Memory)
    int *inputs, *targets;
    float *hidden_batch, *logits_batch;
    float *dW1, *dW2, *db1, *db2;
    float *losses;

    NeuralLM(int vocab_size, int _H, float _lr, int _batch, float _temp, int _top_k = 20)
        : V(vocab_size), H(_H), lr(_lr), batch_size(_batch), temperature(_temp), top_k(_top_k) {

        // Allocate weights in Unified Memory
        CUDA_CHECK(cudaMallocManaged(&W1, V * H * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&b1, H * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&W2, H * V * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&b2, V * sizeof(float)));

        // Random initialize weights
        std::mt19937 gen(42);
        float h_scale = sqrtf(2.0f / V);
        for (int i = 0; i < V * H; i++) W1[i] = ((float)gen() / gen.max() * 2 - 1) * h_scale;
        for (int i = 0; i < H * V; i++) W2[i] = ((float)gen() / gen.max() * 2 - 1) * 0.01f;
        for (int i = 0; i < H; i++) b1[i] = 0.0f;
        for (int i = 0; i < V; i++) b2[i] = 0.0f;

        // Allocate single inference buffers
        CUDA_CHECK(cudaMallocManaged(&in, V * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&hidden, H * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&logits, V * sizeof(float)));

        // Allocate batch training buffers
        size_t maxB = batch_size;
        CUDA_CHECK(cudaMallocManaged(&inputs, maxB * sizeof(int)));
        CUDA_CHECK(cudaMallocManaged(&targets, maxB * sizeof(int)));
        CUDA_CHECK(cudaMallocManaged(&hidden_batch, maxB * H * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&logits_batch, maxB * V * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&dW1, maxB * H * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&dW2, maxB * H * V * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&db1, maxB * H * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&db2, maxB * V * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&losses, maxB * sizeof(float)));
    }

    void destroy_buffers() {
        cudaFree(W1); cudaFree(b1); cudaFree(W2); cudaFree(b2);
        cudaFree(in); cudaFree(hidden); cudaFree(logits);
        cudaFree(inputs); cudaFree(targets);
        cudaFree(hidden_batch); cudaFree(logits_batch);
        cudaFree(dW1); cudaFree(dW2); cudaFree(db1); cudaFree(db2);
        cudaFree(losses);
    }

    ~NeuralLM() {
        destroy_buffers();
    }

    void set_weights(const float* src) {
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

    void forward_gpu_single(int input_char, float* logits_out) {
        for (int j = 0; j < V; j++) in[j] = 0.0f;
        in[input_char] = 1.0f;

        // Perform hidden = in @ W1
        dim3 block(16, 16);
        dim3 grid((H + 15) / 16, 1); // M=1, K=V, N=H
        matmul_kernel<<<grid, block>>>(in, W1, hidden, 1, V, H);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply ReLU and add bias b1
        for (int j = 0; j < H; j++) {
            hidden[j] = std::max(0.0f, hidden[j] + b1[j]);
        }

        // Perform logits = hidden @ W2
        dim3 grid2((V + 15) / 16, 1); // M=1, K=H, N=V
        matmul_kernel<<<grid2, block>>>(hidden, W2, logits, 1, H, V);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Add bias b2
        for (int j = 0; j < V; j++) {
            logits_out[j] = logits[j] + b2[j];
        }
    }

    void train_hybrid(const std::vector<int>& data, int num_cores, std::atomic<bool>& interrupt, int max_epochs = 0) {
        std::cout << "Training (V=" << V << ", H=" << H
                  << ", data=" << data.size() << ", batch=" << batch_size
                  << ", cores=" << num_cores << ")...\n" << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();
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

                for (int i = 0; i < B; i++) {
                    inputs[i] = data[start + i];
                    targets[i] = data[start + i + 1];
                }

                // Launch batch training kernel
                int block_sz = 256;
                int grid_sz = (B + block_sz - 1) / block_sz;
                train_batch_kernel<<<grid_sz, block_sz>>>(
                    inputs, targets, W1, W2, b1, b2,
                    hidden_batch, logits_batch,
                    dW1, dW2, db1, db2, losses, B, V, H
                );

                // Launch batch weight update kernel
                int totalThreads = V * H + H * V + H + V;
                int grid_sz_up = (totalThreads + block_sz - 1) / block_sz;
                train_update_kernel<<<grid_sz_up, block_sz>>>(
                    inputs, dW1, dW2, db1, db2, W1, W2, b1, b2, B, V, H, lr, 1.0f / B
                );

                // Synchronize CPU/GPU
                CUDA_CHECK(cudaDeviceSynchronize());

                for (int i = 0; i < B; i++) {
                    total_loss += losses[i];
                    count++;
                }

                float eta = 0;
                if (max_epochs > 0) {
                    batches_done++;
                    auto now = std::chrono::high_resolution_clock::now();
                    float elapsed = std::chrono::duration<float>(now - t_start).count();
                    float progress = (float)batches_done / total_batches;
                    if (progress > 0) eta = elapsed / progress - elapsed;
                }
                std::string ep_str = max_epochs > 0
                    ? std::to_string(epoch) + "/" + std::to_string(max_epochs)
                    : std::to_string(epoch);
                std::cout << "\rep " << ep_str
                          << " [" << std::string(pct/5, '#') << std::string(20-pct/5, '.') << "]"
                          << " loss=" << std::fixed << std::setprecision(4) << (total_loss / count)
                          << " eta=" << format_eta(eta)
                          << "    " << std::flush;
            }
            std::cout << std::endl;

            if (interrupt) {
                std::cout << "\nInterrupted. Saving model...\n" << std::endl;
                break;
            }
        }
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
        std::cout << "Model loaded from " << path << std::endl;
        return true;
    }

    int sample_next(int input_char, std::mt19937& gen, const std::vector<int>& recent) {
        float* logits_local = new float[V];
        forward_gpu_single(input_char, logits_local);

        float mx = logits_local[0];
        for (int j = 1; j < V; j++) if (logits_local[j] > mx) mx = logits_local[j];

        float* probs = new float[V];
        float se = 0;
        for (int j = 0; j < V; j++) {
            float s = logits_local[j] / temperature;
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
                              [](auto& a, auto& b) { return a.first > b.first; });
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
            if (r < cum) { delete[] logits_local; delete[] probs; return j; }
        }
        delete[] logits_local; delete[] probs;
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
};

// === Parallel Training Coordination ===

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
    std::atomic<bool> coord_ready{false};
    std::atomic<int> coord_ack{0};
};

static void barrier_sense(SharedBarrier* b, int n, bool& local_sense) {
    if (b->b1_count.fetch_add(1) == n - 1) {
        b->b1_count.store(0);
        b->b1_sense.store(local_sense);
    } else {
        while (b->b1_sense.load() != local_sense) {
            if (training_interrupted) std::this_thread::yield();
        }
    }
    local_sense = !local_sense;
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

static void render_lines(int n, int* epochs, float* losses, int* pcts, bool winner, float eta_secs = 0) {
    std::cout << "\033[u";
    for (int w = 0; w < n; w++) {
        int p = std::min(pcts[w], 100);
        std::string bar(20, '.');
        for (int i = 0; i < p / 5; i++) bar[i] = '#';
        std::cout << "\033[2K\033[" << bar_colors[w % 16] << "m[w" << w << "]"
                  << (winner && w == 0 ? " \xe2\x98\x85" : "  ")
                  << "\033[0m ep " << epochs[w]
                  << " [" << bar << "]"
                  << " loss=" << std::fixed << std::setprecision(4) << losses[w]
                  << " eta=" << format_eta(eta_secs)
                  << "\n";
    }
    std::cout << std::flush;
}

static void train_parallel_fork(const std::vector<int>& data, int V, int H,
                                 float lr, int batch_size, int num_cores,
                                 int num_workers, const std::string& model_path,
                                 int max_epochs = 0, bool continue_training = false,
                                 int top_k = 20) {
    size_t wb = NeuralLM::weight_bytes(V, H);
    size_t shm_size = sizeof(SharedBarrier) + sizeof(SharedProgress) + wb * num_workers;
    void* shm = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    SharedBarrier* barrier = (SharedBarrier*)shm;
    SharedProgress* prog = (SharedProgress*)((char*)shm + sizeof(SharedBarrier));
    float* weight_slots = (float*)((char*)shm + sizeof(SharedBarrier) + sizeof(SharedProgress));

    if (continue_training) {
        std::ifstream f(model_path, std::ios::binary);
        if (f) {
            int v, h, bs_s;
            float lr_s, temp_s;
            f.read((char*)&v, sizeof(int));
            f.read((char*)&h, sizeof(int));
            f.read((char*)&lr_s, sizeof(float));
            f.read((char*)&bs_s, sizeof(int));
            f.read((char*)&temp_s, sizeof(float));
            f.read((char*)weight_slots, wb);
            std::cout << "Continuing parallel training from " << model_path << "\n" << std::endl;
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

            // Create NeuralLM on the child context
            NeuralLM model(V, H, lr, batch_size, 0.7f, top_k);
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
                float total_loss = 0;
                int count = 0;
                int batches = 0;
                for (size_t start = 0; start + 1 < slice.size(); start += batch_size) {
                    int B = std::min((size_t)batch_size, slice.size() - 1 - start);
                    for (int i = 0; i < B; i++) {
                        model.inputs[i] = slice[start + i];
                        model.targets[i] = slice[start + i + 1];
                    }

                    int block_sz = 256;
                    int grid_sz = (B + block_sz - 1) / block_sz;
                    train_batch_kernel<<<grid_sz, block_sz>>>(
                        model.inputs, model.targets, model.W1, model.W2, model.b1, model.b2,
                        model.hidden_batch, model.logits_batch,
                        model.dW1, model.dW2, model.db1, model.db2, model.losses, B, V, H
                    );

                    int totalThreads = V * H + H * V + H + V;
                    int grid_sz_up = (totalThreads + block_sz - 1) / block_sz;
                    train_update_kernel<<<grid_sz_up, block_sz>>>(
                        model.inputs, model.dW1, model.dW2, model.db1, model.db2, model.W1, model.W2, model.b1, model.b2, B, V, H, lr, 1.0f / B
                    );

                    CUDA_CHECK(cudaDeviceSynchronize());

                    for (int i = 0; i < B; i++) { total_loss += model.losses[i]; count++; }

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
                            render_lines(num_workers, e_, l_, p_, false, eta);
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
                    render_lines(num_workers, e_, l_, p_, true, eta);
                    barrier->coord_ready.store(true, std::memory_order_release);
                } else {
                    while (!barrier->coord_ready.load(std::memory_order_acquire)) {
                        if (training_interrupted) break;
                    }
                    barrier->coord_ack.fetch_add(1, std::memory_order_acq_rel);
                }

                if (w == 0) {
                    while (barrier->coord_ack.load(std::memory_order_acquire) < num_workers - 1) {
                        if (training_interrupted) break;
                    }
                    barrier->coord_ready.store(false, std::memory_order_release);
                    barrier->coord_ack.store(0, std::memory_order_release);
                }

                memcpy(my_slot, weight_slots, wb);
                model.set_weights(my_slot);

                if (training_interrupted) {
                    if (w == 0) {
                        model.save_model(model_path);
                        std::cout << "\n\033[32m✓ Interrupted. Model saved (best weights).\033[0m\n";
                    }
                    break;
                }
            }

            exit(0);
        } else {
            children[w] = pid;
        }
    }

    for (int w = 0; w < num_workers; w++) waitpid(children[w], nullptr, 0);
    std::cout << "\033[" << num_workers << "B" << std::flush;

    std::ofstream f(model_path, std::ios::binary);
    if (f) {
        int v = V, h = H;
        f.write((const char*)&v, sizeof(int));
        f.write((const char*)&h, sizeof(int));
        float lr_s = lr; f.write((const char*)&lr_s, sizeof(float));
        int bs_s = batch_size; f.write((const char*)&bs_s, sizeof(int));
        float t_s = 0.7f; f.write((const char*)&t_s, sizeof(float));
        f.write((const char*)weight_slots, wb);
        std::cout << "Model saved to " << model_path << std::endl;
    }
    munmap(shm, shm_size);
}

// === Chat Session ===

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
    if (!f) { std::cerr << "Error: cannot open " << path << std::endl; exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
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

int main(int argc, char* argv[]) {
    bool force_train = false;
    int num_cores = std::max(1u, std::thread::hardware_concurrency());
    int n_embd = 256;
    float lr = 0.1f;
    int batch_size = 256;
    float temperature = 0.7f;
    int top_k = 20;
    size_t max_train_chars = 1000000;
    int max_gen_tokens = 500;
    std::string model_path = "model.bin";
    int parallel_workers = 1;
    int max_epochs = 0;
    bool show_model_info = false;
    bool continue_training = false;
    bool want_chat = false;

    auto print_help = [&]() {
        std::cout << "Usage: random_llm [OPTIONS]\n"
                  << "Random neural language model (CUDA accelerated).\n"
                  << "First run auto-trains until Ctrl+C, then chats. Subsequent runs load saved model.\n"
                  << "\nOptions:\n"
                  << "  -t, --train          Force retrain from scratch\n"
                  << "  -c, --core N         CPU threads for gradient reduction (default: max)\n"
                  << "  -e, --embed N        Embedding/hidden size (default: 256)\n"
                  << "  -l, --lr F           Learning rate (default: 0.1)\n"
                  << "  -b, --batch N        Mini-batch size (default: 256)\n"
                  << "  -T, --temp F         Sampling temperature (default: 0.7)\n"
                  << "  -k, --topk N         Top-K sampling (default: 20, 0=disabled)\n"
                  << "  -M, --max-train N    Max training characters (default: 1000000)\n"
                  << "  -n, --max-tokens N   Max generation tokens per reply (default: 500)\n"
                  << "  -m, --model PATH     Model file path (default: model.bin)\n"
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
            model_path = argv[++i];
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
        } else if (a == "--ModelInfo") {
            show_model_info = true;
        }
    }

    if (show_model_info) {
        std::ifstream f(model_path, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open model file: " << model_path << std::endl;
            return 1;
        }
        int v, h, bs_s;
        float lr_s, temp_s;
        f.read((char*)&v, sizeof(int));
        f.read((char*)&h, sizeof(int));
        f.read((char*)&lr_s, sizeof(float));
        f.read((char*)&bs_s, sizeof(int));
        f.read((char*)&temp_s, sizeof(float));
        std::cout << "Model file: " << model_path << "\n"
                  << "  Vocab size (V):    " << v << "\n"
                  << "  Hidden size (H):   " << h << "\n"
                  << "  Parameters:        " << (v * h * 2 + h + v) << "\n"
                  << "  Learning rate:     " << lr_s << "\n"
                  << "  Batch size:        " << bs_s << "\n"
                  << "  Temperature:       " << temp_s << "\n";
        return 0;
    }

    check_ram();

    try {
        std::string corpus = load_corpus("corpus.txt");
        if (corpus.size() > max_train_chars) corpus.resize(max_train_chars);

        Tokenizer tk;
        tk.build(corpus);

        std::vector<int> data;
        for (char c : corpus) data.push_back(tk.stoi[c]);

        std::cout << "\033[1;36m╔══════════════════════════════════════╗\033[0m\n";
        std::cout << "\033[1;36m║      RANDOM NEURAL LANGUAGE MODEL    ║\033[0m\n";
        std::cout << "\033[1;36m╚══════════════════════════════════════╝\033[0m\n";
        std::cout << "Vocab: " << tk.vocab_size << " chars, "
                  << "Params: " << (tk.vocab_size * n_embd * 2 + n_embd + tk.vocab_size) << "\n"
                  << "Train: GPU+CUDA + CPU (" << num_cores << " cores),  "
                  << "Gen: GPU (CUDA MatMul)\n" << std::endl;

        if (parallel_workers > 1) {
            std::cout << "Parallel workers: " << parallel_workers << "\n" << std::endl;
            std::signal(SIGINT, handle_sigint);
            auto t0 = std::chrono::high_resolution_clock::now();
            train_parallel_fork(data, tk.vocab_size, n_embd, lr, batch_size,
                                num_cores, parallel_workers, model_path, max_epochs, continue_training, top_k);
            auto t1 = std::chrono::high_resolution_clock::now();
            float secs = std::chrono::duration<float>(t1 - t0).count();
            std::cout << "\n\033[1;32m✓ Parallel training done in " << std::fixed
                      << std::setprecision(1) << secs << "s\033[0m" << std::endl;

            bool any_training = force_train || continue_training;
            if (want_chat || !any_training) {
                std::cout << "Loading model for chat..." << std::endl;
                std::signal(SIGINT, SIG_DFL);
                NeuralLM model(tk.vocab_size, n_embd, lr, batch_size, temperature, top_k);
                model.load_model(model_path);
                std::cout << std::endl;
                run_chat(model, tk, max_gen_tokens);
            }
        } else {
            NeuralLM model(tk.vocab_size, n_embd, lr, batch_size, temperature, top_k);

            bool should_train = false;
            if (continue_training) {
                if (model.load_model(model_path)) {
                    std::cout << "Continuing training from saved model (" << model_path << ").\n" << std::endl;
                } else {
                    std::cout << "No saved model to continue from. Training from scratch.\n" << std::endl;
                }
                should_train = true;
            } else if (force_train) {
                std::cout << "Force re-train requested (--train).\n" << std::endl;
                should_train = true;
            } else if (!model.load_model(model_path)) {
                std::cout << "No saved model found. Training from scratch.\n" << std::endl;
                should_train = true;
            }

            if (should_train) {
                std::signal(SIGINT, handle_sigint);

                auto t0 = std::chrono::high_resolution_clock::now();
                model.train_hybrid(data, num_cores, training_interrupted, max_epochs);
                auto t1 = std::chrono::high_resolution_clock::now();
                float secs = std::chrono::duration<float>(t1 - t0).count();

                model.save_model(model_path);

                std::cout << "\033[1;32m✓ Trained in " << std::fixed << std::setprecision(1) << secs << "s\033[0m\n" << std::endl;
                std::signal(SIGINT, SIG_DFL);
            }

            if (want_chat || !should_train) {
                run_chat(model, tk, max_gen_tokens);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
