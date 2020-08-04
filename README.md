# Neural Network Multiplication Examples

## Working C++ Implementations

### 1. CPU Version: `neural_network_multiplication.cpp` ⭐
- **Complete implementation** with enhanced features
- **Features:**
  - Xavier/Glorot weight initialization for better learning
  - Input normalization [0,1] improves training stability
  - Learning rate scheduling for optimal convergence
  - Comprehensive evaluation and visualization
- **How to use:**
  ```bash
  g++ -O3 -o neural_network neural_network_multiplication.cpp
  ./neural_network
  ```

### 2. GPU Version: `neural_network_cuda.cu`
- **Working CUDA implementation** (compiles and runs successfully)
- GPU-accelerated neural network with parallel computation
- Avoids header conflicts with custom math functions
- **How to use:**
  ```bash
  nvcc -O2 -o neural_network_cuda neural_network_cuda.cu -lcudart
  ./neural_network_cuda
  ```

## Build System

### CPU Build: `Makefile`
```bash
make                    # Build CPU version
make run               # Build and run CPU version
```

### GPU Build: `Makefile_cuda`
```bash
make -f Makefile_cuda neural_network_multiplication    # Build CPU
make -f Makefile_cuda neural_network_multiplication_cuda  # Build GPU (if headers work)
make -f Makefile_cuda run-gpu                         # Run GPU version
```

## Benchmark Tool

### `benchmark_cpu_gpu.cpp`
- Performance comparison between CPU and GPU implementations
- **How to use:**
  ```bash
  g++ -O3 -o benchmark_cpu_gpu benchmark_cpu_gpu.cpp
  ./benchmark_cpu_gpu
  ```

## Core Files Summary

| File | Type | Status | Purpose |
|------|------|--------|---------|
| `neural_network_multiplication.cpp` | C++ | ✅ Working | CPU implementation with enhanced features |
| `neural_network_cuda.cu` | CUDA | ✅ Working | GPU implementation |
| `benchmark_cpu_gpu.cpp` | C++ | ✅ Working | Performance comparison |
| `Makefile` | Build | ✅ Working | CPU build system |
| `Makefile_cuda` | Build | ✅ Working | GPU build system |

## Quick Start

### Test CPU Learning:
```bash
g++ -O3 neural_network_multiplication.cpp -o neural_network
./neural_network
```

### Test GPU Acceleration:
```bash
nvcc -O2 neural_network_cuda.cu -lcudart -o neural_network_cuda
./neural_network_cuda
```

## Requirements

- **CPU**: GCC/G++ compiler
- **GPU**: NVIDIA GPU with CUDA toolkit
  ```bash
  # Install CUDA (Ubuntu/Debian)
  sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
  ```
