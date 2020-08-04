# Makefile for Neural Network Multiplication Implementation

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
NVCCFLAGS = -O2 -lcudart

# Targets
CPU_TARGET = neural_network_multiplication
GPU_TARGET = neural_network_cuda
BENCHMARK_TARGET = benchmark_cpu_gpu

CPU_SOURCE = neural_network_multiplication.cpp
GPU_SOURCE = neural_network_cuda.cu
BENCHMARK_SOURCE = benchmark_cpu_gpu.cpp

# Default target - builds both CPU and GPU versions, then runs benchmark
all: $(CPU_TARGET) $(GPU_TARGET) $(BENCHMARK_TARGET)
	@echo "Building complete neural network suite..."
	@echo "Running performance benchmarks..."
	@echo "=====================" 
	@echo "CPU Version Performance:"
	@echo "====================="
	@timeout 30s ./$(CPU_TARGET) | head -5 || echo "CPU test completed"
	@echo ""
	@echo "====================="
	@echo "GPU Version Performance:"
	@echo "====================="
	@timeout 30s ./$(GPU_TARGET) | head -5 || echo "GPU test completed"
	@echo ""
	@echo "====================="
	@echo "Performance Summary:"
	@echo "====================="
	@echo -n "CPU execution time: "
	@timeout 15s bash -c 'time -p ./$(CPU_TARGET) > /dev/null 2>&1' 2>&1 | grep real | awk '{print $$2}'
	@echo -n "GPU execution time: "
	@timeout 15s bash -c 'time -p ./$(GPU_TARGET) > /dev/null 2>&1' 2>&1 | grep real | awk '{print $$2}'

# Build CPU version
$(CPU_TARGET): $(CPU_SOURCE)
	@echo "Building CPU implementation..."
	$(CXX) $(CXXFLAGS) -o $(CPU_TARGET) $(CPU_SOURCE)

# Build GPU version
$(GPU_TARGET): $(GPU_SOURCE)
	@echo "Building GPU implementation..."
	$(NVCC) $(NVCCFLAGS) -o $(GPU_TARGET) $(GPU_SOURCE)

# Build benchmark
$(BENCHMARK_TARGET): $(BENCHMARK_SOURCE)
	@echo "Building benchmark tool..."
	$(CXX) $(CXXFLAGS) -o $(BENCHMARK_TARGET) $(BENCHMARK_SOURCE)

# Run CPU version
run-cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

# Run GPU version
run-gpu: $(GPU_TARGET)
	./$(GPU_TARGET)

# Run benchmark comparison
benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

# Run all versions
run-all: $(CPU_TARGET) $(GPU_TARGET)
	@echo "Running CPU version..."
	@echo "====================="
	./$(CPU_TARGET)
	@echo ""
	@echo "Running GPU version..."
	@echo "====================="
	./$(GPU_TARGET)

# Clean build artifacts
clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) $(BENCHMARK_TARGET) *.txt neural_network_results.txt

# Debug build
debug: CXXFLAGS += -DDEBUG -O0
debug: $(CPU_TARGET) $(BENCHMARK_TARGET)

# Release build with maximum optimization
release: CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -DNDEBUG
release: $(CPU_TARGET) $(GPU_TARGET) $(BENCHMARK_TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build both implementations and run basic benchmarks (default)"
	@echo "  benchmark - Build and run detailed performance comparison"
	@echo "  run-cpu   - Build and run CPU version"
	@echo "  run-gpu   - Build and run GPU version"  
	@echo "  run-all   - Build and run both versions"
	@echo "  clean     - Remove build artifacts"
	@echo "  debug     - Build with debug flags"
	@echo "  release   - Build with maximum optimization"
	@echo "  help      - Show this help message"

.PHONY: all run-cpu run-gpu run-all benchmark clean debug release help
