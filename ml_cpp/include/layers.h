/**
 * EchoSurf ML Framework - Neural Network Layers
 *
 * Implements common neural network layers optimized for inference.
 * Based on holistic system approach:
 * - Components (layers) interact to form the Whole (network)
 * - Relations (connections) define the logical world (computation graph)
 * - Energy (data) flows through physical interactions
 */

#ifndef ECHOSURF_LAYERS_H
#define ECHOSURF_LAYERS_H

#include "tensor.h"
#include <string>
#include <memory>
#include <unordered_map>

namespace echosurf {
namespace ml {

/**
 * Activation function types
 */
enum class Activation {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU
};

/**
 * Base Layer class
 *
 * Following the learning loop paradigm:
 * Idea -> Test -> Learn -> Outcome
 * - forward() implements the Idea (prediction)
 * - Outcome feeds back to improve the system
 */
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass (inference)
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    // Layer information
    virtual std::string name() const = 0;
    virtual size_t param_count() const = 0;

    // Weight access for loading
    virtual void set_weights(const std::unordered_map<std::string, Tensor>& weights) = 0;

    // Output shape computation
    virtual TensorShape output_shape(const TensorShape& input_shape) const = 0;
};

/**
 * Dense (Fully Connected) Layer
 *
 * Implements: output = activation(input @ weights + bias)
 *
 * Sensorimotor coordination inspired:
 * - Weights represent synaptic strengths
 * - Bias represents baseline activation
 * - Activation provides non-linearity for learning
 */
class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_features, size_t output_features,
               Activation activation = Activation::None,
               bool use_bias = true);

    void forward(const Tensor& input, Tensor& output) override;

    std::string name() const override { return name_; }
    size_t param_count() const override;

    void set_weights(const std::unordered_map<std::string, Tensor>& weights) override;

    TensorShape output_shape(const TensorShape& input_shape) const override;

    // Direct weight setters (for model loading)
    void set_kernel(const Tensor& kernel);
    void set_bias(const Tensor& bias);

    // Weight access
    const Tensor& kernel() const { return weights_; }
    const Tensor& bias() const { return bias_; }

    // Configuration
    size_t input_features() const { return input_features_; }
    size_t output_features() const { return output_features_; }
    Activation activation() const { return activation_; }

private:
    std::string name_;
    size_t input_features_;
    size_t output_features_;
    Activation activation_;
    bool use_bias_;

    Tensor weights_;  // Shape: (input_features, output_features)
    Tensor bias_;     // Shape: (output_features,)

    // Pre-allocated buffers for inference (avoid allocations)
    mutable Tensor linear_output_;
    mutable Tensor activation_output_;

    void apply_activation(Tensor& x);
};

/**
 * Dropout Layer
 *
 * During inference: no-op (identity function)
 * This layer exists for model structure compatibility
 */
class DropoutLayer : public Layer {
public:
    explicit DropoutLayer(float rate = 0.5f) : rate_(rate) {}

    void forward(const Tensor& input, Tensor& output) override {
        // Inference mode: just copy
        if (output.shape() != input.shape()) {
            output = Tensor(input.shape());
        }
        std::copy(input.data(), input.data() + input.size(), output.data());
    }

    std::string name() const override { return "dropout"; }
    size_t param_count() const override { return 0; }

    void set_weights(const std::unordered_map<std::string, Tensor>&) override {}

    TensorShape output_shape(const TensorShape& input_shape) const override {
        return input_shape;
    }

    float rate() const { return rate_; }

private:
    float rate_;
};

/**
 * Batch Normalization Layer
 *
 * Normalizes inputs for stable training
 * During inference: uses stored running statistics
 */
class BatchNormLayer : public Layer {
public:
    BatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f);

    void forward(const Tensor& input, Tensor& output) override;

    std::string name() const override { return "batch_norm"; }
    size_t param_count() const override { return num_features_ * 4; }

    void set_weights(const std::unordered_map<std::string, Tensor>& weights) override;

    TensorShape output_shape(const TensorShape& input_shape) const override {
        return input_shape;
    }

    void set_gamma(const Tensor& gamma) { gamma_ = gamma; }
    void set_beta(const Tensor& beta) { beta_ = beta; }
    void set_running_mean(const Tensor& mean) { running_mean_ = mean; }
    void set_running_var(const Tensor& var) { running_var_ = var; }

private:
    size_t num_features_;
    float epsilon_;
    float momentum_;

    Tensor gamma_;        // Scale parameter
    Tensor beta_;         // Shift parameter
    Tensor running_mean_; // Running mean statistics
    Tensor running_var_;  // Running variance statistics
};

/**
 * Flatten Layer
 *
 * Converts multi-dimensional input to 1D
 */
class FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    void forward(const Tensor& input, Tensor& output) override;

    std::string name() const override { return "flatten"; }
    size_t param_count() const override { return 0; }

    void set_weights(const std::unordered_map<std::string, Tensor>&) override {}

    TensorShape output_shape(const TensorShape& input_shape) const override {
        return TensorShape({input_shape.size()});
    }
};

/**
 * Conv2D Layer
 *
 * 2D Convolution for visual processing models
 * Implements spatial pattern detection
 */
class Conv2DLayer : public Layer {
public:
    Conv2DLayer(size_t in_channels, size_t out_channels,
                size_t kernel_size, size_t stride = 1,
                size_t padding = 0, Activation activation = Activation::None);

    void forward(const Tensor& input, Tensor& output) override;

    std::string name() const override { return "conv2d"; }
    size_t param_count() const override;

    void set_weights(const std::unordered_map<std::string, Tensor>& weights) override;

    TensorShape output_shape(const TensorShape& input_shape) const override;

    void set_kernel(const Tensor& kernel) { kernel_ = kernel; }
    void set_bias(const Tensor& bias) { bias_ = bias; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    Activation activation_;

    Tensor kernel_;  // Shape: (out_channels, in_channels, kernel_size, kernel_size)
    Tensor bias_;    // Shape: (out_channels,)
};

/**
 * MaxPool2D Layer
 *
 * Spatial downsampling for visual features
 */
class MaxPool2DLayer : public Layer {
public:
    explicit MaxPool2DLayer(size_t pool_size = 2, size_t stride = 0);

    void forward(const Tensor& input, Tensor& output) override;

    std::string name() const override { return "maxpool2d"; }
    size_t param_count() const override { return 0; }

    void set_weights(const std::unordered_map<std::string, Tensor>&) override {}

    TensorShape output_shape(const TensorShape& input_shape) const override;

private:
    size_t pool_size_;
    size_t stride_;
};

/**
 * Activation Layer (standalone)
 *
 * Applies activation function independently
 */
class ActivationLayer : public Layer {
public:
    explicit ActivationLayer(Activation type) : type_(type) {}

    void forward(const Tensor& input, Tensor& output) override;

    std::string name() const override { return activation_name(); }
    size_t param_count() const override { return 0; }

    void set_weights(const std::unordered_map<std::string, Tensor>&) override {}

    TensorShape output_shape(const TensorShape& input_shape) const override {
        return input_shape;
    }

private:
    Activation type_;

    std::string activation_name() const;
};

} // namespace ml
} // namespace echosurf

#endif // ECHOSURF_LAYERS_H
