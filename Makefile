# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall
LIBS = -lvulkan

# Shader compiler
GLSL_COMPILER = glslangValidator
GLSL_FLAGS = -V

# Folders
SHADERS_DIR = vulkan_llm/shaders
SRC_DIR = vulkan_llm

# Files
COMP_SHADERS = $(SHADERS_DIR)/matmul.comp $(SHADERS_DIR)/train_batch.comp $(SHADERS_DIR)/train_update.comp
SPIRV_SHADERS = $(SHADERS_DIR)/matmul.spv $(SHADERS_DIR)/train_batch.spv $(SHADERS_DIR)/train_update.spv
MAIN_SRC = $(SRC_DIR)/main.cpp
MAIN_BIN = $(SRC_DIR)/vulkan_llm

all: $(SPIRV_SHADERS) $(MAIN_BIN)

$(SHADERS_DIR)/%.spv: $(SHADERS_DIR)/%.comp
	@echo "Compiling shader: $< -> $@"
	$(GLSL_COMPILER) $(GLSL_FLAGS) $< -o $@

$(MAIN_BIN): $(MAIN_SRC) $(SRC_DIR)/vulkan_compute.hpp $(SPIRV_SHADERS)
	@echo "Compiling C++ application: $< -> $@"
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

test: all
	@echo "Running Vulkan LLM Test..."
	cd $(SRC_DIR) && ../$(MAIN_BIN)

clean:
	rm -f $(SPIRV_SHADERS) $(MAIN_BIN)

.PHONY: all test clean
