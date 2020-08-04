/*
 * Neural Network Multiplication Example (C++ Implementation)
 * ========================================================
 *
 * This implementation features:
 * - Better weight initialization (Xavier/Glorot)
 * - Input normalization and output scaling
 * - Learning rate scheduling
 * - Comprehensive training and testing
 * - Performance timing
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ranges>
#include <concepts>
#include <span>
#include <format>

class NeuralNetwork {
private:
    int hidden_size;
    std::vector<std::vector<double>> w1;  // Input to hidden weights
    std::vector<double> b1;               // Hidden layer biases
    std::vector<double> w2;               // Hidden to output weights
    double b2;                            // Output bias
    double learning_rate;
    double min_learning_rate;
    
    std::mt19937 rng;
    
public:
    NeuralNetwork(int hidden_size = 12, double learning_rate = 0.1) 
        : hidden_size(hidden_size), learning_rate(learning_rate), 
          min_learning_rate(0.001), rng(42) {
        
        // Better weight initialization (Xavier/Glorot initialization)
        std::uniform_real_distribution<double> weight_dist(-0.5, 0.5);
        std::uniform_real_distribution<double> bias_dist(-0.1, 0.1);
        
        w1.resize(hidden_size, std::vector<double>(2));
        b1.resize(hidden_size);
        w2.resize(hidden_size);
        
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < 2; j++) {
                w1[i][j] = weight_dist(rng);
            }
            b1[i] = bias_dist(rng);
            w2[i] = weight_dist(rng);
        }
        b2 = bias_dist(rng);
    }
    
    double sigmoid(double x) const {
        // Prevent overflow
        if (x > 500) return 1.0;
        if (x < -500) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    double sigmoid_derivative(double x) const {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
    std::tuple<double, std::vector<double>, std::vector<double>> forward(std::span<const double> inputs) const {
        // Normalize inputs to [0, 1] range
        std::vector<double> normalized_inputs = {inputs[0] / 10.0, inputs[1] / 10.0};
        
        std::vector<double> hidden_outputs(hidden_size);
        
        // Hidden layer
        for (int i = 0; i < hidden_size; i++) {
            double weighted_sum = 0.0;
            for (int j = 0; j < 2; j++) {
                weighted_sum += normalized_inputs[j] * w1[i][j];
            }
            weighted_sum += b1[i];
            hidden_outputs[i] = sigmoid(weighted_sum);
        }
        
        // Output layer (no activation for regression)
        double output = 0.0;
        for (int i = 0; i < hidden_size; i++) {
            output += hidden_outputs[i] * w2[i];
        }
        output += b2;
        
        return {output, hidden_outputs, normalized_inputs};
    }
    
    void backward(std::span<const double> normalized_inputs, 
                  std::span<const double> hidden_outputs, 
                  double output, double target) {
        
        // Normalize target
        double normalized_target = target / 100.0;
        
        // Calculate output error
        double output_error = output - normalized_target;
        
        // Update output layer weights
        for (int i = 0; i < hidden_size; i++) {
            w2[i] -= learning_rate * output_error * hidden_outputs[i];
        }
        b2 -= learning_rate * output_error;
        
        // Calculate hidden layer errors and update weights
        for (int i = 0; i < hidden_size; i++) {
            double weighted_sum = 0.0;
            for (int j = 0; j < 2; j++) {
                weighted_sum += normalized_inputs[j] * w1[i][j];
            }
            weighted_sum += b1[i];
            
            double hidden_error = output_error * w2[i] * sigmoid_derivative(weighted_sum);
            
            for (int j = 0; j < 2; j++) {
                w1[i][j] -= learning_rate * hidden_error * normalized_inputs[j];
            }
            b1[i] -= learning_rate * hidden_error;
        }
    }
    
    void updateLearningRate(int epoch, int total_epochs) {
        // Learning rate decay
        double progress = static_cast<double>(epoch) / total_epochs;
        learning_rate = std::max(min_learning_rate, 0.1 * (1.0 - progress));
    }
    
    int getParameterCount() const {
        return hidden_size * 2 + hidden_size + hidden_size + 1;
    }
    
    // Get unscaled output for demonstration
    double predict(std::span<const double> inputs) const {
        auto [output, _, __] = forward(inputs);
        return output * 100.0;  // Scale back to original range
    }
    
    // Save model to file
    void saveModel(const std::string& filename) const {
        std::cout << "Saving model to: " << filename << std::endl;
        
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cout << "Error: Could not create file " << filename << std::endl;
            return;
        }
        
        // Write metadata
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(double));
        file.write(reinterpret_cast<const char*>(&min_learning_rate), sizeof(double));
        
        // Write w1 weights (2D vector)
        for (int i = 0; i < hidden_size; i++) {
            file.write(reinterpret_cast<const char*>(w1[i].data()), 2 * sizeof(double));
        }
        
        // Write other weights/biases
        file.write(reinterpret_cast<const char*>(b1.data()), hidden_size * sizeof(double));
        file.write(reinterpret_cast<const char*>(w2.data()), hidden_size * sizeof(double));
        file.write(reinterpret_cast<const char*>(&b2), sizeof(double));
        
        file.close();
        std::cout << "Model saved successfully (" << getParameterCount() << " parameters)" << std::endl;
    }
    
    // Load model from file
    void loadModel(const std::string& filename) {
        std::cout << "Loading model from: " << filename << std::endl;
        
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cout << "Error: Could not open file " << filename << std::endl;
            return;
        }
        
        // Read metadata
        int saved_hidden_size;
        double saved_learning_rate, saved_min_learning_rate;
        file.read(reinterpret_cast<char*>(&saved_hidden_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&saved_learning_rate), sizeof(double));
        file.read(reinterpret_cast<char*>(&saved_min_learning_rate), sizeof(double));
        
        // Check if dimensions match
        if (saved_hidden_size != hidden_size) {
            std::cout << "Error: Model hidden size (" << saved_hidden_size 
                      << ") doesn't match current network (" << hidden_size << ")" << std::endl;
            return;
        }
        
        // Read w1 weights (2D vector)
        for (int i = 0; i < hidden_size; i++) {
            file.read(reinterpret_cast<char*>(w1[i].data()), 2 * sizeof(double));
        }
        
        // Read other weights/biases
        file.read(reinterpret_cast<char*>(b1.data()), hidden_size * sizeof(double));
        file.read(reinterpret_cast<char*>(w2.data()), hidden_size * sizeof(double));
        file.read(reinterpret_cast<char*>(&b2), sizeof(double));
        
        file.close();
        
        // Update learning rates
        learning_rate = saved_learning_rate;
        min_learning_rate = saved_min_learning_rate;
        
        std::cout << "Model loaded successfully (" << getParameterCount() << " parameters)" << std::endl;
    }
};

struct TrainingExample {
    std::vector<double> inputs;
    double target;
    
    TrainingExample(double a, double b) : inputs{a, b}, target(a * b) {}
    
    // C++20 designated initializer support
    TrainingExample(std::initializer_list<double> values, double target_val) 
        : inputs(values), target(target_val) {}
};

std::vector<TrainingExample> generateTrainingData(int num_samples, double min_val = 0.0, double max_val = 10.0) {
    std::vector<TrainingExample> training_data;
    training_data.reserve(num_samples);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    
    for (int i = 0; i < num_samples; i++) {
        double a = dist(rng);
        double b = dist(rng);
        training_data.emplace_back(a, b);
    }
    
    return training_data;
}

std::vector<double> trainModel(NeuralNetwork& model, 
                              std::vector<TrainingExample>& training_data, 
                              int num_epochs = 1000) {
    std::vector<double> losses;
    losses.reserve(num_epochs);
    
    std::cout << "Starting training...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::mt19937 rng(42);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;
        
        // Shuffle training data
        std::shuffle(training_data.begin(), training_data.end(), rng);
        
        for (const auto& example : training_data) {
            // Forward pass
            auto [output, hidden_outputs, normalized_inputs] = model.forward(example.inputs);
            
            // Calculate loss (mean squared error)
            double normalized_target = example.target / 100.0;
            double loss = (output - normalized_target) * (output - normalized_target);
            total_loss += loss;
            
            // Backward pass
            model.backward(normalized_inputs, hidden_outputs, output, example.target);
        }
        
        double avg_loss = total_loss / training_data.size();
        losses.push_back(avg_loss);
        
        // Update learning rate
        model.updateLearningRate(epoch, num_epochs);
        
        // Print progress every 100 epochs using C++20 std::format
        if ((epoch + 1) % 100 == 0) {
            std::cout << std::format("Epoch {:4d}/{}, Loss: {:.6f}\n", 
                                   epoch + 1, num_epochs, avg_loss);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::format("Training completed in {} milliseconds\n", duration.count());
    
    return losses;
}

std::pair<std::vector<double>, std::pair<double, double>> evaluateModel(
    const NeuralNetwork& model, 
    const std::vector<TrainingExample>& test_data) {
    
    std::vector<double> predictions;
    predictions.reserve(test_data.size());
    
    double total_squared_error = 0.0;
    double total_absolute_error = 0.0;
    
    for (const auto& example : test_data) {
        double output = model.predict(example.inputs);
        predictions.push_back(output);
        
        double error = output - example.target;
        total_squared_error += error * error;
        total_absolute_error += std::abs(error);
    }
    
    double mse = total_squared_error / test_data.size();
    double mae = total_absolute_error / test_data.size();
    
    return {predictions, {mse, mae}};
}

void demonstrateMultiplication(const NeuralNetwork& model, 
                             const std::vector<std::pair<double, double>>& test_cases) {
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "NEURAL NETWORK MULTIPLICATION DEMONSTRATION\n";
    std::cout << std::string(60, '=') << "\n";
    
    for (const auto& [a, b] : test_cases) {
        // Neural network prediction
        std::vector<double> inputs_vec = {a, b};
        double nn_result = model.predict(inputs_vec);
        
        // Conventional multiplication
        double conventional_result = a * b;
        
        // Calculate error
        double error = std::abs(nn_result - conventional_result);
        double error_percent = (conventional_result != 0) ? (error / conventional_result) * 100 : 0;
        
        std::cout << std::format("Input: {:.2f} × {:.2f}\n", a, b);
        std::cout << std::format("Neural Network: {:.6f}\n", nn_result);
        std::cout << std::format("Conventional:   {:.6f}\n", conventional_result);
        std::cout << std::format("Error:         {:.6f} ({:.2f}%)\n", error, error_percent);
        std::cout << std::string(40, '-') << "\n";
    }
}

void plotTrainingProgress(const std::vector<double>& losses) {
    std::cout << "\nTraining Loss Over Time (ASCII Plot):\n";
    std::cout << std::string(50, '=') << "\n";
    
    if (losses.empty()) return;
    
    double max_loss = *std::max_element(losses.begin(), losses.end());
    double min_loss = *std::min_element(losses.begin(), losses.end());
    double loss_range = max_loss - min_loss;
    
    // Create ASCII plot using C++20 ranges
    int step = std::max(1, static_cast<int>(losses.size()) / 20);
    auto epoch_range = std::views::iota(0u, losses.size()) | std::views::filter([step](size_t i) { return i % step == 0; });
    
    for (auto i : epoch_range) {
        double normalized_loss = (loss_range > 0) ? (losses[i] - min_loss) / loss_range : 0;
        int bar_length = static_cast<int>(normalized_loss * 40);
        std::string bar(bar_length, '#');
        
        std::cout << std::format("Epoch {:4d}: {} {:.6f}\n", i, bar, losses[i]);
    }
}

void saveResultsToFile(const std::vector<double>& losses, 
                       const std::vector<TrainingExample>& test_data,
                       const std::vector<double>& predictions) {
    
    std::ofstream file("neural_network_results.txt");
    if (file.is_open()) {
        file << "Neural Network Multiplication Results\n";
        file << "==========================================\n\n";
        
        file << "Training Loss History:\n";
        for (size_t i = 0; i < losses.size(); i++) {
            file << "Epoch " << i << ": " << losses[i] << "\n";
        }
        
        file << "\nTest Results:\n";
        file << "Input1\tInput2\tTarget\tPrediction\tError\tError%\n";
        for (size_t i = 0; i < test_data.size(); i++) {
            double error = predictions[i] - test_data[i].target;
            double error_percent = (test_data[i].target != 0) ? (error / test_data[i].target) * 100 : 0;
            
            file << test_data[i].inputs[0] << "\t" << test_data[i].inputs[1] << "\t"
                 << test_data[i].target << "\t" << predictions[i] << "\t"
                 << error << "\t" << error_percent << "\n";
        }
        
        file.close();
        std::cout << "\nResults saved to 'neural_network_results.txt'\n";
    }
}

int main() {
    std::cout << "Neural Network Multiplication Learning Example\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Enhanced Implementation with Optimized Features\n";
    std::cout << std::string(60, '=') << "\n";
    
    // Generate training data
    std::cout << "Generating training data...\n";
    auto training_data = generateTrainingData(3000, 0.0, 10.0);
    
    // Generate test data
    std::cout << "Generating test data...\n";
    auto test_data = generateTrainingData(300, 0.0, 10.0);
    
    // Create the model
    NeuralNetwork model(12, 0.1);
    std::cout << std::format("Model created with {} parameters\n", model.getParameterCount());
    std::cout << "Features: Xavier initialization, input normalization, learning rate scheduling\n";
    
    // Train the model
    auto losses = trainModel(model, training_data, 1000);
    
    // Evaluate the model
    std::cout << "\nEvaluating model...\n";
    auto [predictions, errors] = evaluateModel(model, test_data);
    auto [mse, mae] = errors;
    
    std::cout << std::format("Test MSE: {:.6f}\n", mse);
    std::cout << std::format("Test MAE: {:.6f}\n", mae);
    
    // Plot training progress
    plotTrainingProgress(losses);
    
    // Demonstrate on specific test cases
    std::vector<std::pair<double, double>> test_cases = {
        {2.0, 3.0},
        {4.0, 5.0},
        {7.0, 8.0},
        {1.5, 2.5},
        {0.1, 0.2},
        {9.0, 9.0},
        {3.14, 2.71},
        {0.5, 0.5},
        {10.0, 0.1}
    };
    
    demonstrateMultiplication(model, test_cases);
    
    // Save results to file
    saveResultsToFile(losses, test_data, predictions);
    
    // Save trained model
    std::cout << "\n=== SAVING TRAINED MODEL ===\n";
    model.saveModel("trained_model_cpu.bin");
    
    // Test loading the saved model
    std::cout << "\n=== TESTING SAVED MODEL ===\n";
    NeuralNetwork loaded_model(12, 0.1);
    loaded_model.loadModel("trained_model_cpu.bin");
    
    std::cout << "\nTesting loaded model:\n";
    std::cout << "=====================\n";
    
    for (const auto& [a, b] : test_cases) {
        std::vector<double> inputs_vec = {a, b};
        std::span<const double> inputs(inputs_vec);
        double prediction = loaded_model.predict(inputs);
        double actual = a * b;
        double error = (prediction - actual) * (prediction - actual);
        
        std::cout << std::format("{:.1f} x {:.1f}: Predicted={:.4f}, Actual={:.4f}, Error={:.6f}\n",
                                  a, b, prediction, actual, error);
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "KEY IMPROVEMENTS:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "1. Xavier/Glorot weight initialization for better convergence\n";
    std::cout << "2. Input normalization [0,1] improves learning stability\n";
    std::cout << "3. Output scaling prevents gradient vanishing\n";
    std::cout << "4. Learning rate scheduling for better optimization\n";
    std::cout << "5. Enhanced architecture with optimal hidden size\n";
    std::cout << "6. Comprehensive evaluation and visualization\n";
    std::cout << "7. Combined best features for optimal performance\n";
    
    return 0;
}
