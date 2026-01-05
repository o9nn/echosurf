/**
 * EchoSurf ML Framework - Neural Network Layers Implementation
 */

#include "layers.h"
#include <cstring>
#include <algorithm>

namespace echosurf {
namespace ml {

// ============================================================================
// DenseLayer Implementation
// ============================================================================

DenseLayer::DenseLayer(size_t input_features, size_t output_features,
                       Activation activation, bool use_bias)
    : name_("dense")
    , input_features_(input_features)
    , output_features_(output_features)
    , activation_(activation)
    , use_bias_(use_bias)
    , weights_(TensorShape({input_features, output_features}))
    , bias_(TensorShape({output_features}))
    , linear_output_(TensorShape({1, output_features}))
    , activation_output_(TensorShape({1, output_features}))
{
    // Initialize weights to zero (will be loaded from model)
    weights_.zero();
    bias_.zero();
}

void DenseLayer::forward(const Tensor& input, Tensor& output) {
    // Determine batch size
    size_t batch_size = 1;
    size_t features = input.size();

    if (input.ndim() == 2) {
        batch_size = input.shape()[0];
        features = input.shape()[1];
    }

    if (features != input_features_) {
        throw std::invalid_argument("DenseLayer: input feature size mismatch");
    }

    // Reshape input to 2D if needed
    Tensor input_2d = input.ndim() == 2 ? input :
        input.reshape(TensorShape({batch_size, features}));

    // Prepare output
    TensorShape out_shape({batch_size, output_features_});
    if (output.shape() != out_shape) {
        output = Tensor(out_shape);
    }

    // Matrix multiplication: output = input @ weights
    ops::matmul(input_2d, weights_, output);

    // Add bias
    if (use_bias_) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < output_features_; ++j) {
                output.at(b, j) += bias_[j];
            }
        }
    }

    // Apply activation
    apply_activation(output);
}

void DenseLayer::apply_activation(Tensor& x) {
    switch (activation_) {
        case Activation::None:
            break;
        case Activation::ReLU:
            ops::relu_inplace(x);
            break;
        case Activation::Sigmoid:
            ops::sigmoid_inplace(x);
            break;
        case Activation::Tanh:
            ops::tanh_inplace(x);
            break;
        case Activation::Softmax:
            {
                Tensor temp(x.shape());
                ops::softmax(x, temp);
                std::copy(temp.data(), temp.data() + temp.size(), x.data());
            }
            break;
        case Activation::LeakyReLU:
            {
                Tensor temp(x.shape());
                ops::leaky_relu(x, temp, 0.01f);
                std::copy(temp.data(), temp.data() + temp.size(), x.data());
            }
            break;
    }
}

size_t DenseLayer::param_count() const {
    size_t count = input_features_ * output_features_;
    if (use_bias_) {
        count += output_features_;
    }
    return count;
}

void DenseLayer::set_weights(const std::unordered_map<std::string, Tensor>& weights) {
    auto kernel_it = weights.find("kernel");
    if (kernel_it != weights.end()) {
        set_kernel(kernel_it->second);
    }

    auto bias_it = weights.find("bias");
    if (bias_it != weights.end()) {
        set_bias(bias_it->second);
    }
}

void DenseLayer::set_kernel(const Tensor& kernel) {
    if (kernel.size() != weights_.size()) {
        throw std::invalid_argument("Kernel size mismatch");
    }
    std::copy(kernel.data(), kernel.data() + kernel.size(), weights_.data());
}

void DenseLayer::set_bias(const Tensor& bias) {
    if (bias.size() != bias_.size()) {
        throw std::invalid_argument("Bias size mismatch");
    }
    std::copy(bias.data(), bias.data() + bias.size(), bias_.data());
}

TensorShape DenseLayer::output_shape(const TensorShape& input_shape) const {
    if (input_shape.ndim() == 1) {
        return TensorShape({output_features_});
    } else {
        return TensorShape({input_shape[0], output_features_});
    }
}

// ============================================================================
// BatchNormLayer Implementation
// ============================================================================

BatchNormLayer::BatchNormLayer(size_t num_features, float epsilon, float momentum)
    : num_features_(num_features)
    , epsilon_(epsilon)
    , momentum_(momentum)
    , gamma_(TensorShape({num_features}), 1.0f)
    , beta_(TensorShape({num_features}), 0.0f)
    , running_mean_(TensorShape({num_features}), 0.0f)
    , running_var_(TensorShape({num_features}), 1.0f)
{}

void BatchNormLayer::forward(const Tensor& input, Tensor& output) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    const size_t batch_size = input.ndim() == 2 ? input.shape()[0] : 1;
    const size_t features = input.ndim() == 2 ? input.shape()[1] : input.size();

    // Inference mode: use running statistics
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            size_t idx = b * features + f;
            float normalized = (input[idx] - running_mean_[f]) /
                              std::sqrt(running_var_[f] + epsilon_);
            output[idx] = gamma_[f] * normalized + beta_[f];
        }
    }
}

void BatchNormLayer::set_weights(const std::unordered_map<std::string, Tensor>& weights) {
    auto gamma_it = weights.find("gamma");
    if (gamma_it != weights.end()) {
        set_gamma(gamma_it->second);
    }

    auto beta_it = weights.find("beta");
    if (beta_it != weights.end()) {
        set_beta(beta_it->second);
    }

    auto mean_it = weights.find("moving_mean");
    if (mean_it != weights.end()) {
        set_running_mean(mean_it->second);
    }

    auto var_it = weights.find("moving_variance");
    if (var_it != weights.end()) {
        set_running_var(var_it->second);
    }
}

// ============================================================================
// FlattenLayer Implementation
// ============================================================================

void FlattenLayer::forward(const Tensor& input, Tensor& output) {
    TensorShape flat_shape({input.size()});
    if (output.shape() != flat_shape) {
        output = Tensor(flat_shape);
    }
    std::copy(input.data(), input.data() + input.size(), output.data());
}

// ============================================================================
// Conv2DLayer Implementation
// ============================================================================

Conv2DLayer::Conv2DLayer(size_t in_channels, size_t out_channels,
                         size_t kernel_size, size_t stride,
                         size_t padding, Activation activation)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , activation_(activation)
    , kernel_(TensorShape({out_channels, in_channels, kernel_size, kernel_size}))
    , bias_(TensorShape({out_channels}))
{
    kernel_.zero();
    bias_.zero();
}

void Conv2DLayer::forward(const Tensor& input, Tensor& output) {
    // Input shape: (batch, channels, height, width) or (channels, height, width)
    // For simplicity, assume input is flattened or handle 4D case

    // This is a simplified implementation for inference
    // For production, use im2col optimization or call optimized library

    if (input.ndim() < 3) {
        throw std::invalid_argument("Conv2D requires at least 3D input");
    }

    const size_t batch = input.ndim() == 4 ? input.shape()[0] : 1;
    const size_t in_c = input.ndim() == 4 ? input.shape()[1] : input.shape()[0];
    const size_t in_h = input.ndim() == 4 ? input.shape()[2] : input.shape()[1];
    const size_t in_w = input.ndim() == 4 ? input.shape()[3] : input.shape()[2];

    const size_t out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    const size_t out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;

    TensorShape out_shape({batch, out_channels_, out_h, out_w});
    if (output.shape() != out_shape) {
        output = Tensor(out_shape);
    }
    output.zero();

    // Naive convolution (for correctness; optimize later)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = bias_[oc];

                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        for (size_t kh = 0; kh < kernel_size_; ++kh) {
                            for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                int ih = static_cast<int>(oh * stride_ + kh) - static_cast<int>(padding_);
                                int iw = static_cast<int>(ow * stride_ + kw) - static_cast<int>(padding_);

                                if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                    iw >= 0 && iw < static_cast<int>(in_w)) {
                                    size_t in_idx = b * in_c * in_h * in_w +
                                                   ic * in_h * in_w +
                                                   static_cast<size_t>(ih) * in_w +
                                                   static_cast<size_t>(iw);
                                    size_t k_idx = oc * in_channels_ * kernel_size_ * kernel_size_ +
                                                  ic * kernel_size_ * kernel_size_ +
                                                  kh * kernel_size_ + kw;
                                    sum += input[in_idx] * kernel_[k_idx];
                                }
                            }
                        }
                    }

                    size_t out_idx = b * out_channels_ * out_h * out_w +
                                    oc * out_h * out_w +
                                    oh * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    // Apply activation
    switch (activation_) {
        case Activation::ReLU:
            ops::relu_inplace(output);
            break;
        case Activation::Sigmoid:
            ops::sigmoid_inplace(output);
            break;
        default:
            break;
    }
}

size_t Conv2DLayer::param_count() const {
    return out_channels_ * in_channels_ * kernel_size_ * kernel_size_ + out_channels_;
}

void Conv2DLayer::set_weights(const std::unordered_map<std::string, Tensor>& weights) {
    auto kernel_it = weights.find("kernel");
    if (kernel_it != weights.end()) {
        set_kernel(kernel_it->second);
    }

    auto bias_it = weights.find("bias");
    if (bias_it != weights.end()) {
        set_bias(bias_it->second);
    }
}

TensorShape Conv2DLayer::output_shape(const TensorShape& input_shape) const {
    size_t batch = input_shape.ndim() == 4 ? input_shape[0] : 1;
    size_t in_h = input_shape.ndim() == 4 ? input_shape[2] : input_shape[1];
    size_t in_w = input_shape.ndim() == 4 ? input_shape[3] : input_shape[2];

    size_t out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    size_t out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;

    return TensorShape({batch, out_channels_, out_h, out_w});
}

// ============================================================================
// MaxPool2DLayer Implementation
// ============================================================================

MaxPool2DLayer::MaxPool2DLayer(size_t pool_size, size_t stride)
    : pool_size_(pool_size)
    , stride_(stride == 0 ? pool_size : stride)
{}

void MaxPool2DLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() < 3) {
        throw std::invalid_argument("MaxPool2D requires at least 3D input");
    }

    const size_t batch = input.ndim() == 4 ? input.shape()[0] : 1;
    const size_t channels = input.ndim() == 4 ? input.shape()[1] : input.shape()[0];
    const size_t in_h = input.ndim() == 4 ? input.shape()[2] : input.shape()[1];
    const size_t in_w = input.ndim() == 4 ? input.shape()[3] : input.shape()[2];

    const size_t out_h = (in_h - pool_size_) / stride_ + 1;
    const size_t out_w = (in_w - pool_size_) / stride_ + 1;

    TensorShape out_shape({batch, channels, out_h, out_w});
    if (output.shape() != out_shape) {
        output = Tensor(out_shape);
    }

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float max_val = std::numeric_limits<float>::lowest();

                    for (size_t ph = 0; ph < pool_size_; ++ph) {
                        for (size_t pw = 0; pw < pool_size_; ++pw) {
                            size_t ih = oh * stride_ + ph;
                            size_t iw = ow * stride_ + pw;

                            size_t in_idx = b * channels * in_h * in_w +
                                           c * in_h * in_w +
                                           ih * in_w + iw;
                            max_val = std::max(max_val, input[in_idx]);
                        }
                    }

                    size_t out_idx = b * channels * out_h * out_w +
                                    c * out_h * out_w +
                                    oh * out_w + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

TensorShape MaxPool2DLayer::output_shape(const TensorShape& input_shape) const {
    size_t batch = input_shape.ndim() == 4 ? input_shape[0] : 1;
    size_t channels = input_shape.ndim() == 4 ? input_shape[1] : input_shape[0];
    size_t in_h = input_shape.ndim() == 4 ? input_shape[2] : input_shape[1];
    size_t in_w = input_shape.ndim() == 4 ? input_shape[3] : input_shape[2];

    size_t out_h = (in_h - pool_size_) / stride_ + 1;
    size_t out_w = (in_w - pool_size_) / stride_ + 1;

    return TensorShape({batch, channels, out_h, out_w});
}

// ============================================================================
// ActivationLayer Implementation
// ============================================================================

void ActivationLayer::forward(const Tensor& input, Tensor& output) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    switch (type_) {
        case Activation::ReLU:
            ops::relu(input, output);
            break;
        case Activation::Sigmoid:
            ops::sigmoid(input, output);
            break;
        case Activation::Tanh:
            ops::tanh_activation(input, output);
            break;
        case Activation::Softmax:
            ops::softmax(input, output);
            break;
        case Activation::LeakyReLU:
            ops::leaky_relu(input, output);
            break;
        default:
            std::copy(input.data(), input.data() + input.size(), output.data());
            break;
    }
}

std::string ActivationLayer::activation_name() const {
    switch (type_) {
        case Activation::ReLU: return "relu";
        case Activation::Sigmoid: return "sigmoid";
        case Activation::Tanh: return "tanh";
        case Activation::Softmax: return "softmax";
        case Activation::LeakyReLU: return "leaky_relu";
        default: return "linear";
    }
}

} // namespace ml
} // namespace echosurf
