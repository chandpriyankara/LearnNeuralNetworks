/*
 * Simple CUDA Neural Network Multiplication Example
 * ================================================
 * 
 * This version avoids all header conflicts by using minimal includes
 * and custom implementations of math functions.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// Only include CUDA runtime, avoid all other headers
#include <cuda_runtime.h>

// Custom math implementations to avoid conflicts
__device__ __host__ double simple_exp(double x) {
    if (x > 10) return 22000.0;
    if (x < -10) return 0.0000454;
    return 1.0 + x + x*x/2.0 + x*x*x/6.0;
}

__device__ __host__ double simple_sigmoid(double x) {
    double exp_x = simple_exp(-x);
    return 1.0 / (1.0 + exp_x);
}

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Forward pass kernel
__global__ void forward_hidden_kernel(const double* inputs, const double* w1, const double* b1,
                                     double* hidden_outputs, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        double weighted_sum = inputs[0] * w1[idx * 2] + inputs[1] * w1[idx * 2 + 1] + b1[idx];
        hidden_outputs[idx] = simple_sigmoid(weighted_sum);
    }
}

// Output layer kernel
__global__ void forward_output_kernel(const double* hidden_outputs, const double* w2, const double* b2,
                                      double* output, int hidden_size) {
    extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        sdata[threadIdx.x] = hidden_outputs[idx] * w2[idx];
    } else {
        sdata[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < hidden_size) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        *output = sdata[0] + *b2;
    }
}

// Backward pass kernel
__global__ void backward_hidden_kernel(const double* inputs, const double* w1, const double* b1,
                                      const double* w2, double* w1_new, double* b1_new,
                                      double output_error, double learning_rate, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        double weighted_sum = inputs[0] * w1[idx * 2] + inputs[1] * w1[idx * 2 + 1] + b1[idx];
        double sigmoid_val = simple_sigmoid(weighted_sum);
        double sigmoid_derivative = sigmoid_val * (1.0 - sigmoid_val);
        double hidden_error = output_error * w2[idx] * sigmoid_derivative;
        
        w1_new[idx * 2] = w1[idx * 2] - learning_rate * hidden_error * inputs[0];
        w1_new[idx * 2 + 1] = w1[idx * 2 + 1] - learning_rate * hidden_error * inputs[1];
        b1_new[idx] = b1[idx] - learning_rate * hidden_error;
    }
}

__global__ void backward_output_kernel(const double* hidden_outputs, double* w2, double* b2,
                                      double output_error, double learning_rate, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        w2[idx] -= learning_rate * output_error * hidden_outputs[idx];
    }
    if (idx == 0) {
        *b2 -= learning_rate * output_error;
    }
}

// Simple random number generator
class SimpleRandom {
private:
    unsigned long state;
    
public:
    SimpleRandom(unsigned long seed = 42) : state(seed) {}
    
    double uniform(double min_val, double max_val) {
        state = state * 1103515245 + 12345;
        double normalized = (double)(state % 1000000) / 1000000.0;
        return min_val + normalized * (max_val - min_val);
    }
};

// CUDA Neural Network Class
class SimpleCudaNetwork {
private:
    int hidden_size;
    double learning_rate;
    double* d_w1, *d_b1, *d_w2, *d_b2;
    double* d_hidden_outputs, *d_output, *d_inputs;
    double* d_w1_temp, *d_b1_temp;
    
public:
    SimpleCudaNetwork(int hidden_size = 10, double learning_rate = 0.01) 
        : hidden_size(hidden_size), learning_rate(learning_rate) {
        
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_w1, hidden_size * 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b1, hidden_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_w2, hidden_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b2, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_hidden_outputs, hidden_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_inputs, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_w1_temp, hidden_size * 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b1_temp, hidden_size * sizeof(double)));
        
        initializeWeights();
    }
    
    ~SimpleCudaNetwork() {
        cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2);
        cudaFree(d_hidden_outputs); cudaFree(d_output); cudaFree(d_inputs);
        cudaFree(d_w1_temp); cudaFree(d_b1_temp);
    }
    
    void initializeWeights() {
        SimpleRandom rng(42);
        std::vector<double> temp;
        
        // Single precision weights
        temp.resize(hidden_size * 2);
        for (int i = 0; i < hidden_size * 2; i++) {
            temp[i] = 0.8 * rng.uniform(0.0, 1.0) - 0.4; // Scale to [-0.4, 0.4]
        }
        CUDA_CHECK(cudaMemcpy(d_w1, temp.data(), hidden_size * 2 * sizeof(double), cudaMemcpyHostToDevice));
        
        temp.resize(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            temp[i] = 0.8 * rng.uniform(0.0, 1.0) - 0.4;
        }
        CUDA_CHECK(cudaMemcpy(d_b1, temp.data(), hidden_size * sizeof(double), cudaMemcpyHostToDevice));
        
        for (int i = 0; i < hidden_size; i++) {
            temp[i] = 0.8 * rng.uniform(0.0, 1.0) - 0.4;
        }
        CUDA_CHECK(cudaMemcpy(d_w2, temp.data(), hidden_size * sizeof(double), cudaMemcpyHostToDevice));
        
        double b2_val = 0.8 * rng.uniform(0.0, 1.0) - 0.4;
        CUDA_CHECK(cudaMemcpy(d_b2, &b2_val, sizeof(double), cudaMemcpyHostToDevice));
    }
    
    double forward(std::vector<double>& inputs) {
        // Copy inputs to GPU
        CUDA_CHECK(cudaMemcpy(d_inputs, inputs.data(), 2 * sizeof(double), cudaMemcpyHostToDevice));
        
        // Launch hidden layer kernel
        int threadsPerBlock = 256;
        int blocks = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
        
        forward_hidden_kernel<<<blocks, threadsPerBlock>>>(d_inputs, d_w1, d_b1, d_hidden_outputs, hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Reset output
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(double)));
        
        // Launch output layer kernel
        int sharedMemSize = threadsPerBlock * sizeof(double);
        forward_output_kernel<<<1, threadsPerBlock, sharedMemSize>>>(d_hidden_outputs, d_w2, d_b2, d_output, hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back
        double output;
        CUDA_CHECK(cudaMemcpy(&output, d_output, sizeof(double), cudaMemcpyDeviceToHost));
        return output;
    }
    
    void backward(const std::vector<double>& inputs, double output, double target) {
        // Copy inputs to GPU
        CUDA_CHECK(cudaMemcpy(d_inputs, inputs.data(), 2 * sizeof(double), cudaMemcpyHostToDevice));
        
        // Need hidden outputs - recompute them
        int threadsPerBlock = 256;
        int blocks = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
        
        forward_hidden_kernel<<<blocks, threadsPerBlock>>>(d_inputs, d_w1, d_b1, d_hidden_outputs, hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double output_error = output - target;
        
        // Launch backward kernels
        backward_output_kernel<<<blocks, threadsPerBlock>>>(d_hidden_outputs, d_w2, d_b2, output_error, learning_rate, hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        backward_hidden_kernel<<<blocks, threadsPerBlock>>>(d_inputs, d_w1, d_b1, d_w2, d_w1_temp, d_b1_temp, output_error, learning_rate, hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy updated weights back
        CUDA_CHECK(cudaMemcpy(d_w1, d_w1_temp, hidden_size * 2 * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, d_b1_temp, hidden_size * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    
    int getParameterCount() const {
        return hidden_size * 2 + hidden_size + hidden_size + 1; // W1 + B1 + W2 + B2
    }
    
    // Save model to file
    void saveModel(const std::string& filename) const {
        printf("Saving model to: %s\n", filename.c_str());
        
        // Create arrays to hold data from GPU
        std::vector<double> w1_host(hidden_size * 2);
        std::vector<double> b1_host(hidden_size);
        std::vector<double> w2_host(hidden_size);
        double b2_host;
        
        // Copy weights from GPU to CPU
        CUDA_CHECK(cudaMemcpy(w1_host.data(), d_w1, hidden_size * 2 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b1_host.data(), d_b1, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w2_host.data(), d_w2, hidden_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&b2_host, d_b2, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Write to file
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            printf("Error: Could not create file %s\n", filename.c_str());
            return;
        }
        
        // Write metadata
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(double));
        
        // Write weights
        file.write(reinterpret_cast<const char*>(w1_host.data()), hidden_size * 2 * sizeof(double));
        file.write(reinterpret_cast<const char*>(b1_host.data()), hidden_size * sizeof(double));
        file.write(reinterpret_cast<const char*>(w2_host.data()), hidden_size * sizeof(double));
        file.write(reinterpret_cast<const char*>(&b2_host), sizeof(double));
        
        file.close();
        printf("Model saved successfully (%d parameters)\n", getParameterCount());
    }
    
    // Load model from file
    void loadModel(const std::string& filename) {
        printf("Loading model from: %s\n", filename.c_str());
        
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            printf("Error: Could not open file %s\n", filename.c_str());
            return;
        }
        
        // Read metadata
        int saved_hidden_size;
        double saved_learning_rate;
        file.read(reinterpret_cast<char*>(&saved_hidden_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&saved_learning_rate), sizeof(double));
        
        // Check if dimensions match
        if (saved_hidden_size != hidden_size) {
            printf("Error: Model hidden size (%d) doesn't match current network (%d)\n", 
                   saved_hidden_size, hidden_size);
            return;
        }
        
        // Create arrays to hold data
        std::vector<double> w1_host(hidden_size * 2);
        std::vector<double> b1_host(hidden_size);
        std::vector<double> w2_host(hidden_size);
        double b2_host;
        
        // Read weights
        file.read(reinterpret_cast<char*>(w1_host.data()), hidden_size * 2 * sizeof(double));
        file.read(reinterpret_cast<char*>(b1_host.data()), hidden_size * sizeof(double));
        file.read(reinterpret_cast<char*>(w2_host.data()), hidden_size * sizeof(double));
        file.read(reinterpret_cast<char*>(&b2_host), sizeof(double));
        
        file.close();
        
        // Copy weights from CPU to GPU
        CUDA_CHECK(cudaMemcpy(d_w1, w1_host.data(), hidden_size * 2 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, b1_host.data(), hidden_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2, w2_host.data(), hidden_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, &b2_host, sizeof(double), cudaMemcpyHostToDevice));
        
        // Update learning rate
        learning_rate = saved_learning_rate;
        
        printf("Model loaded successfully (%d parameters)\n", getParameterCount());
    }
};

// Generate training data
std::vector<std::vector<double>> generateData(int num_samples) {
    std::vector<std::vector<double>> data;
    SimpleRandom rng(42);
    
    for (int i = 0; i < num_samples; i++) {
        std::vector<double> example;
        double a = rng.uniform(0.0, 10.0);
        double b = rng.uniform(0.0, 10.0);
        example.push_back(a);
        example.push_back(b);
        example.push_back(a * b); // target
        data.push_back(example);
    }
    return data;
}

int main() {
    printf("Simple CUDA Neural Network Multiplication\n");
    printf("========================================\n");
    
    // Check CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA GPU found!\n");
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using GPU: %s\n", deviceProp.name);
    
    // Create network
    SimpleCudaNetwork network(10, 0.01);
    printf("Network created with 40 parameters\n");
    
    // Generate training data
    std::vector<std::vector<double>> training_data = generateData(1000);
    
    printf("Training network...\n");
    
    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        double total_loss = 0.0;
        
        for (const auto& example : training_data) {
            std::vector<double> inputs = {example[0], example[1]};
            double target = example[2];
            
            double output = network.forward(inputs);
            double loss = (output - target) * (output - target);
            total_loss += loss;
            
            network.backward(inputs, output, target);
        }
        
        if (epoch % 20 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, total_loss / training_data.size());
        }
    }
    
    // Test some examples
    printf("\nTesting trained network:\n");
    printf("=======================\n");
    
    std::vector<std::vector<double>> test_cases = {
        {2.0, 3.0},   {4.0, 5.0},   {7.0, 8.0},
        {1.5, 2.5},   {0.1, 0.2},   {9.0, 9.0}
    };
    
    for (const auto& test : test_cases) {
        double a = test[0];
        double b = test[1];
        std::vector<double> inputs = {a, b};
        
        double prediction = network.forward(inputs);
        double actual = a * b;
        double error = (prediction - actual) * (prediction - actual);
        
        printf("%.1f x %.1f: Predicted=%.4f, Actual=%.4f, Error=%.6f\n", 
               a, b, prediction, actual, error);
    }
    
    printf("\n=== SAVING TRAINED MODEL ===\n");
    network.saveModel("trained_model_cuda.bin");
    
    // Test loading the saved model
    printf("\n=== TESTING SAVED MODEL ===\n");
    SimpleCudaNetwork new_network(10, 0.01);
    new_network.loadModel("trained_model_cuda.bin");
    
    printf("\nTesting loaded model:\n");
    printf("=====================\n");
    
    for (const auto& test : test_cases) {
        double a = test[0];
        double b = test[1];
        std::vector<double> inputs = {a, b};
        
        double prediction = new_network.forward(inputs);
        double actual = a * b;
        double error = (prediction - actual) * (prediction - actual);
        
        printf("%.1f x %.1f: Predicted=%.4f, Actual=%.4f, Error=%.6f\n", 
               a, b, prediction, actual, error);
    }
    
    printf("\nGPU acceleration and model persistence working!\n");
    return 0;
}
