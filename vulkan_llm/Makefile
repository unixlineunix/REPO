CXX      = g++
CXXFLAGS = -std=c++11 -O2
LDFLAGS  = -pthread -lvulkan

all: random_llm

random_llm: main.cpp vulkan_compute.hpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

clean:
	rm -f random_llm

.PHONY: all clean
