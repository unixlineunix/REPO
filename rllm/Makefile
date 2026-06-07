# Compiler and flags
NVCC = /opt/cuda/bin/nvcc
NVCCFLAGS = -std=c++17 -O3 -Xcompiler -Wall

# Files
MAIN_SRC = main.cu
MAIN_BIN = random_llm

all: $(MAIN_BIN)

$(MAIN_BIN): $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

test: all
	@echo "Running Random CUDA LLM Test..."
	./$(MAIN_BIN)

clean:
	rm -f $(MAIN_BIN)

.PHONY: all test clean
