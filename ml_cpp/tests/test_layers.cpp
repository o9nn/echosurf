/**
 * EchoSurf ML Framework - Layer Tests
 */

#include "layers.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

using namespace echosurf::ml;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " << #name << "... "; \
    test_##name(); \
    std::cout << "PASS\n"; \
} while(0)

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))

TEST(dense_layer_creation) {
    DenseLayer layer(10, 5, Activation::ReLU, true);

    ASSERT_EQ(layer.input_features(), 10);
    ASSERT_EQ(layer.output_features(), 5);
    ASSERT_EQ(layer.param_count(), 55);  // 10*5 + 5
}

TEST(dense_layer_forward) {
    DenseLayer layer(4, 3, Activation::None, true);

    // Set identity-like weights for testing
    Tensor kernel(TensorShape({4, 3}));
    kernel.zero();
    kernel.at(0, 0) = 1.0f;
    kernel.at(1, 1) = 1.0f;
    kernel.at(2, 2) = 1.0f;

    Tensor bias(TensorShape({3}));
    bias[0] = 0.1f;
    bias[1] = 0.2f;
    bias[2] = 0.3f;

    layer.set_kernel(kernel);
    layer.set_bias(bias);

    // Input: [1, 2, 3, 4]
    Tensor input(TensorShape({1, 4}), std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output;

    layer.forward(input, output);

    // Expected: [1*1 + 0.1, 2*1 + 0.2, 3*1 + 0.3] = [1.1, 2.2, 3.3]
    ASSERT_NEAR(output[0], 1.1f, 1e-5f);
    ASSERT_NEAR(output[1], 2.2f, 1e-5f);
    ASSERT_NEAR(output[2], 3.3f, 1e-5f);
}

TEST(dense_layer_relu) {
    DenseLayer layer(2, 2, Activation::ReLU, true);

    Tensor kernel(TensorShape({2, 2}));
    kernel[0] = 1.0f;
    kernel[1] = 0.0f;
    kernel[2] = 0.0f;
    kernel[3] = 1.0f;

    Tensor bias(TensorShape({2}));
    bias[0] = -1.0f;  // Will make first output negative
    bias[1] = 1.0f;   // Will keep second output positive

    layer.set_kernel(kernel);
    layer.set_bias(bias);

    Tensor input(TensorShape({1, 2}), std::vector<float>{0.5f, 0.5f});
    Tensor output;

    layer.forward(input, output);

    // Before ReLU: [0.5 - 1.0, 0.5 + 1.0] = [-0.5, 1.5]
    // After ReLU: [0, 1.5]
    ASSERT_NEAR(output[0], 0.0f, 1e-5f);
    ASSERT_NEAR(output[1], 1.5f, 1e-5f);
}

TEST(dense_layer_softmax) {
    DenseLayer layer(3, 3, Activation::Softmax, false);

    // Identity kernel
    Tensor kernel(TensorShape({3, 3}));
    kernel.zero();
    kernel.at(0, 0) = 1.0f;
    kernel.at(1, 1) = 1.0f;
    kernel.at(2, 2) = 1.0f;

    layer.set_kernel(kernel);

    Tensor input(TensorShape({1, 3}), std::vector<float>{1.0f, 2.0f, 3.0f});
    Tensor output;

    layer.forward(input, output);

    // Output should sum to 1
    float sum = output[0] + output[1] + output[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    // Largest input -> largest probability
    assert(output[2] > output[1]);
    assert(output[1] > output[0]);
}

TEST(dropout_layer) {
    DropoutLayer layer(0.5f);

    ASSERT_NEAR(layer.rate(), 0.5f, 1e-6f);
    ASSERT_EQ(layer.param_count(), 0);

    Tensor input(TensorShape({4}), std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output;

    // In inference mode, dropout is identity
    layer.forward(input, output);

    ASSERT_NEAR(output[0], 1.0f, 1e-6f);
    ASSERT_NEAR(output[1], 2.0f, 1e-6f);
    ASSERT_NEAR(output[2], 3.0f, 1e-6f);
    ASSERT_NEAR(output[3], 4.0f, 1e-6f);
}

TEST(batch_norm_layer) {
    BatchNormLayer layer(4);

    // Set running statistics
    Tensor mean(TensorShape({4}), std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f});
    Tensor var(TensorShape({4}), std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gamma(TensorShape({4}), std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    Tensor beta(TensorShape({4}), std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});

    layer.set_running_mean(mean);
    layer.set_running_var(var);
    layer.set_gamma(gamma);
    layer.set_beta(beta);

    // Input with values equal to running mean should output ~0
    Tensor input(TensorShape({1, 4}), std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f});
    Tensor output;

    layer.forward(input, output);

    // Normalized: (x - mean) / sqrt(var + eps) * gamma + beta
    // All should be ~0 since input == mean and gamma=1, beta=0
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(output[i], 0.0f, 1e-4f);
    }
}

TEST(flatten_layer) {
    FlattenLayer layer;

    Tensor input(TensorShape({2, 3, 4}));
    for (size_t i = 0; i < 24; ++i) {
        input[i] = static_cast<float>(i);
    }

    Tensor output;
    layer.forward(input, output);

    ASSERT_EQ(output.ndim(), 1);
    ASSERT_EQ(output.size(), 24);

    // Values should be preserved
    for (size_t i = 0; i < 24; ++i) {
        ASSERT_NEAR(output[i], static_cast<float>(i), 1e-6f);
    }
}

TEST(activation_layer) {
    // ReLU
    {
        ActivationLayer layer(Activation::ReLU);
        Tensor input(TensorShape({4}), std::vector<float>{-1.0f, 0.0f, 1.0f, 2.0f});
        Tensor output;
        layer.forward(input, output);

        ASSERT_NEAR(output[0], 0.0f, 1e-6f);
        ASSERT_NEAR(output[1], 0.0f, 1e-6f);
        ASSERT_NEAR(output[2], 1.0f, 1e-6f);
        ASSERT_NEAR(output[3], 2.0f, 1e-6f);
    }

    // Sigmoid
    {
        ActivationLayer layer(Activation::Sigmoid);
        Tensor input(TensorShape({3}), std::vector<float>{-10.0f, 0.0f, 10.0f});
        Tensor output;
        layer.forward(input, output);

        ASSERT_NEAR(output[0], 0.0f, 0.01f);
        ASSERT_NEAR(output[1], 0.5f, 1e-6f);
        ASSERT_NEAR(output[2], 1.0f, 0.01f);
    }

    // Tanh
    {
        ActivationLayer layer(Activation::Tanh);
        Tensor input(TensorShape({3}), std::vector<float>{-10.0f, 0.0f, 10.0f});
        Tensor output;
        layer.forward(input, output);

        ASSERT_NEAR(output[0], -1.0f, 0.01f);
        ASSERT_NEAR(output[1], 0.0f, 1e-6f);
        ASSERT_NEAR(output[2], 1.0f, 0.01f);
    }
}

TEST(dense_layer_batch) {
    DenseLayer layer(3, 2, Activation::None, true);

    // Simple weights
    Tensor kernel(TensorShape({3, 2}));
    kernel[0] = 1.0f; kernel[1] = 0.0f;
    kernel[2] = 0.0f; kernel[3] = 1.0f;
    kernel[4] = 1.0f; kernel[5] = 1.0f;

    Tensor bias(TensorShape({2}));
    bias[0] = 0.0f;
    bias[1] = 0.0f;

    layer.set_kernel(kernel);
    layer.set_bias(bias);

    // Batch of 2 inputs
    Tensor input(TensorShape({2, 3}));
    input.at(0, 0) = 1.0f; input.at(0, 1) = 2.0f; input.at(0, 2) = 3.0f;
    input.at(1, 0) = 4.0f; input.at(1, 1) = 5.0f; input.at(1, 2) = 6.0f;

    Tensor output;
    layer.forward(input, output);

    // First sample: [1+3, 2+3] = [4, 5]
    ASSERT_NEAR(output.at(0, 0), 4.0f, 1e-5f);
    ASSERT_NEAR(output.at(0, 1), 5.0f, 1e-5f);

    // Second sample: [4+6, 5+6] = [10, 11]
    ASSERT_NEAR(output.at(1, 0), 10.0f, 1e-5f);
    ASSERT_NEAR(output.at(1, 1), 11.0f, 1e-5f);
}

int main() {
    std::cout << "=== EchoSurf ML Layer Tests ===\n\n";

    RUN_TEST(dense_layer_creation);
    RUN_TEST(dense_layer_forward);
    RUN_TEST(dense_layer_relu);
    RUN_TEST(dense_layer_softmax);
    RUN_TEST(dropout_layer);
    RUN_TEST(batch_norm_layer);
    RUN_TEST(flatten_layer);
    RUN_TEST(activation_layer);
    RUN_TEST(dense_layer_batch);

    std::cout << "\nAll layer tests passed!\n";
    return 0;
}
