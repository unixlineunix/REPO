#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cstring>

class VulkanCompute {
public:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;

    // Matmul pipeline (existing)
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptorSetLayout;

    // Training pipelines
    struct PipelineResources {
        VkDescriptorSetLayout descLayout;
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
    };

    PipelineResources trainBatchRes;
    PipelineResources trainUpdateRes;

    struct Params {
        uint32_t M, K, N;
    };

    struct TrainBatchParams {
        int32_t B, V, H;
    };

    struct TrainUpdateParams {
        int32_t B, V, H;
        float lr, invB;
    };

    VulkanCompute() {
        // 1. Create Instance
        VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pApplicationInfo = &appInfo;
        vkCreateInstance(&createInfo, nullptr, &instance);

        // 2. Pick Physical Device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physicalDevice = devices[0];

        // 3. Create Logical Device
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueCreateInfo.queueFamilyIndex = 0;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        vkGetDeviceQueue(device, 0, 0, &queue);

        // 4. Command Pool
        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = 0;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

        // 5. Descriptor Pool (generous for all pipelines)
        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 50};
        VkDescriptorPoolCreateInfo descPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        descPoolInfo.maxSets = 20;
        descPoolInfo.poolSizeCount = 1;
        descPoolInfo.pPoolSizes = &poolSize;
        descPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &descriptorPool);

        initPipelines();
    }

    VkShaderModule loadShaderModule(const std::string& spvPath) {
        std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
        if (!file) { std::cerr << "Error: cannot open " << spvPath << std::endl; return VK_NULL_HANDLE; }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        VkShaderModuleCreateInfo shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = size;
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
        VkShaderModule shaderModule;
        vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule);
        return shaderModule;
    }

    VkDescriptorSetLayout createDescriptorSetLayout(const VkDescriptorSetLayoutBinding* bindings, uint32_t count) {
        VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = count;
        layoutInfo.pBindings = bindings;
        VkDescriptorSetLayout layout;
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout);
        return layout;
    }

    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout descLayout, uint32_t pushConstantSize) {
        VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descLayout;
        pipelineLayoutInfo.pushConstantRangeCount = pushConstantSize > 0 ? 1 : 0;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantSize > 0 ? &pushConstantRange : nullptr;
        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &layout);
        return layout;
    }

    VkPipeline createComputePipeline(VkShaderModule module, VkPipelineLayout layout) {
        VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = layout;
        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        return pipeline;
    }

    PipelineResources createPipelineResources(const std::string& spvPath,
                                              const VkDescriptorSetLayoutBinding* bindings,
                                              uint32_t bindingCount,
                                              uint32_t pushConstantSize) {
        PipelineResources res;
        res.descLayout = createDescriptorSetLayout(bindings, bindingCount);
        res.pipelineLayout = createPipelineLayout(res.descLayout, pushConstantSize);
        VkShaderModule mod = loadShaderModule(spvPath);
        res.pipeline = createComputePipeline(mod, res.pipelineLayout);
        vkDestroyShaderModule(device, mod, nullptr);
        return res;
    }

    void initPipelines() {
        // === Matmul pipeline (existing, 3 bindings) ===
        VkDescriptorSetLayoutBinding matmulBindings[3] = {};
        for (int i = 0; i < 3; i++) {
            matmulBindings[i].binding = i;
            matmulBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            matmulBindings[i].descriptorCount = 1;
            matmulBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        descriptorSetLayout = createDescriptorSetLayout(matmulBindings, 3);
        pipelineLayout = createPipelineLayout(descriptorSetLayout, sizeof(Params));

        VkShaderModule matmulModule = loadShaderModule("shaders/matmul.spv");
        pipeline = createComputePipeline(matmulModule, pipelineLayout);
        vkDestroyShaderModule(device, matmulModule, nullptr);

        // === Train Batch pipeline (13 bindings) ===
        VkDescriptorSetLayoutBinding trainBatchBindings[13];
        for (int i = 0; i < 13; i++) {
            trainBatchBindings[i].binding = i;
            trainBatchBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            trainBatchBindings[i].descriptorCount = 1;
            trainBatchBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        trainBatchRes = createPipelineResources("shaders/train_batch.spv",
                                                 trainBatchBindings, 13, sizeof(TrainBatchParams));

        // === Train Update pipeline (9 bindings) ===
        VkDescriptorSetLayoutBinding trainUpdateBindings[9];
        for (int i = 0; i < 9; i++) {
            trainUpdateBindings[i].binding = i;
            trainUpdateBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            trainUpdateBindings[i].descriptorCount = 1;
            trainUpdateBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        trainUpdateRes = createPipelineResources("shaders/train_update.spv",
                                                  trainUpdateBindings, 9, sizeof(TrainUpdateParams));
    }

    void dispatchCompute(PipelineResources& res, VkBuffer* buffers, uint32_t bufferCount,
                         void* pushConstants, uint32_t pushSize,
                         uint32_t gx, uint32_t gy, uint32_t gz) {
        VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &res.descLayout;
        VkDescriptorSet descriptorSet;
        vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

        std::vector<VkDescriptorBufferInfo> bufferInfos(bufferCount);
        std::vector<VkWriteDescriptorSet> writes(bufferCount);
        for (uint32_t i = 0; i < bufferCount; i++) {
            bufferInfos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, bufferCount, writes.data(), 0, nullptr);

        VkCommandBufferAllocateInfo cmdAllocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, res.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushConstants);
        vkCmdDispatch(commandBuffer, gx, gy, gz);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    }

    void executeMatMul(VkBuffer A, VkBuffer B, VkBuffer C, uint32_t M, uint32_t K, uint32_t N) {
        VkBuffer bufs[3] = {A, B, C};
        Params p = {M, K, N};
        PipelineResources res = {descriptorSetLayout, pipelineLayout, pipeline};
        dispatchCompute(res, bufs, 3, &p, sizeof(Params), (N + 15) / 16, (M + 15) / 16, 1);
    }

    void executeTrainBatch(VkBuffer inputBuf, VkBuffer targetBuf,
                           VkBuffer w1Buf, VkBuffer w2Buf, VkBuffer b1Buf, VkBuffer b2Buf,
                           VkBuffer hiddenBuf, VkBuffer logitsBuf,
                           VkBuffer dW1Buf, VkBuffer dW2Buf, VkBuffer db1Buf, VkBuffer db2Buf,
                           VkBuffer lossBuf, int B, int V, int H) {
        VkBuffer bufs[13] = {inputBuf, targetBuf, w1Buf, w2Buf, b1Buf, b2Buf,
                              hiddenBuf, logitsBuf, dW1Buf, dW2Buf, db1Buf, db2Buf, lossBuf};
        TrainBatchParams p = {B, V, H};
        dispatchCompute(trainBatchRes, bufs, 13, &p, sizeof(TrainBatchParams), B, 1, 1);
    }

    void executeTrainUpdate(VkBuffer inputBuf,
                            VkBuffer dW1Buf, VkBuffer dW2Buf, VkBuffer db1Buf, VkBuffer db2Buf,
                            VkBuffer w1Buf, VkBuffer w2Buf, VkBuffer b1Buf, VkBuffer b2Buf,
                            int B, int V, int H, float lr) {
        VkBuffer bufs[9] = {inputBuf, dW1Buf, dW2Buf, db1Buf, db2Buf, w1Buf, w2Buf, b1Buf, b2Buf};
        TrainUpdateParams p = {B, V, H, lr, 1.0f / B};
        int totalThreads = V * H + H * V + H + V;
        dispatchCompute(trainUpdateRes, bufs, 9, &p, sizeof(TrainUpdateParams), totalThreads, 1, 1);
    }

    void createBuffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, buffer, &memReqs);
        VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &allocInfo, nullptr, &memory);
        vkBindBufferMemory(device, buffer, memory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
        }
        return 0;
    }

    ~VulkanCompute() {
        vkDestroyPipeline(device, trainBatchRes.pipeline, nullptr);
        vkDestroyPipelineLayout(device, trainBatchRes.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, trainBatchRes.descLayout, nullptr);
        vkDestroyPipeline(device, trainUpdateRes.pipeline, nullptr);
        vkDestroyPipelineLayout(device, trainUpdateRes.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, trainUpdateRes.descLayout, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};
