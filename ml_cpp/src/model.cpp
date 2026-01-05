/**
 * EchoSurf ML Framework - Model Implementation
 */

#include "model.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace echosurf {
namespace ml {

// ============================================================================
// SequentialModel Implementation
// ============================================================================

void SequentialModel::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
    intermediate_outputs_.resize(layers_.size());
}

Tensor SequentialModel::forward(const Tensor& input) {
    if (layers_.empty()) {
        return input.copy();
    }

    // First layer
    layers_[0]->forward(input, intermediate_outputs_[0]);

    // Remaining layers
    for (size_t i = 1; i < layers_.size(); ++i) {
        layers_[i]->forward(intermediate_outputs_[i - 1], intermediate_outputs_[i]);
    }

    return intermediate_outputs_.back().copy();
}

Tensor SequentialModel::predict(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();

    Tensor output = forward(input);

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    metrics_.update(time_ms);

    return output;
}

Tensor SequentialModel::predict_batch(const std::vector<Tensor>& inputs) {
    if (inputs.empty()) {
        return Tensor();
    }

    // For now, process sequentially (TODO: true batching)
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());

    for (const auto& input : inputs) {
        outputs.push_back(predict(input));
    }

    // Concatenate outputs
    size_t output_size = outputs[0].size();
    Tensor batch_output(TensorShape({inputs.size(), output_size}));

    for (size_t i = 0; i < outputs.size(); ++i) {
        std::copy(outputs[i].data(), outputs[i].data() + output_size,
                 batch_output.data() + i * output_size);
    }

    return batch_output;
}

size_t SequentialModel::total_params() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        total += layer->param_count();
    }
    return total;
}

TensorShape SequentialModel::output_shape(const TensorShape& input_shape) const {
    if (layers_.empty()) {
        return input_shape;
    }

    TensorShape shape = input_shape;
    for (const auto& layer : layers_) {
        shape = layer->output_shape(shape);
    }
    return shape;
}

std::string SequentialModel::summary() const {
    std::ostringstream oss;
    oss << "Model: " << (name_.empty() ? "Sequential" : name_) << "\n";
    oss << std::string(60, '=') << "\n";
    oss << std::left << std::setw(25) << "Layer"
        << std::setw(20) << "Output Shape"
        << std::right << std::setw(15) << "Params" << "\n";
    oss << std::string(60, '-') << "\n";

    size_t total = 0;
    for (const auto& layer : layers_) {
        size_t params = layer->param_count();
        total += params;
        oss << std::left << std::setw(25) << layer->name()
            << std::setw(20) << "..."
            << std::right << std::setw(15) << params << "\n";
    }

    oss << std::string(60, '=') << "\n";
    oss << "Total params: " << total << "\n";
    oss << "\nPerformance metrics:\n";
    oss << "  Avg inference: " << std::fixed << std::setprecision(3)
        << metrics_.avg_inference_ms << " ms\n";
    oss << "  Min/Max: " << metrics_.min_inference_ms << " / "
        << metrics_.max_inference_ms << " ms\n";
    oss << "  Total inferences: " << metrics_.total_inferences << "\n";

    return oss.str();
}

// ============================================================================
// ReflexModel Implementation
// ============================================================================

ReflexModel::ReflexModel() : model_("ReflexModel") {
    // Build architecture matching Python implementation
    // Input: 8 features
    // Dense(128) -> ReLU -> Dropout(0.1)
    // Dense(64) -> ReLU -> Dropout(0.1)
    // Dense(32) -> ReLU
    // Dense(4) -> Softmax

    model_.add<DenseLayer>(8, 128, Activation::ReLU);
    model_.add<DropoutLayer>(0.1f);
    model_.add<DenseLayer>(128, 64, Activation::ReLU);
    model_.add<DropoutLayer>(0.1f);
    model_.add<DenseLayer>(64, 32, Activation::ReLU);
    model_.add<DenseLayer>(32, 4, Activation::Softmax);
}

ReflexModel::Action ReflexModel::predict(const ReflexInput& input) {
    Tensor probs = predict_probs(input);

    // Find argmax
    int best_action = 0;
    float best_prob = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_action = static_cast<int>(i);
        }
    }

    return static_cast<Action>(best_action);
}

Tensor ReflexModel::predict_probs(const ReflexInput& input) {
    Tensor input_tensor = input.to_tensor();
    return model_.predict(input_tensor.reshape(TensorShape({1, 8})));
}

Tensor ReflexModel::forward(const Tensor& input) {
    return model_.forward(input);
}

// ============================================================================
// TacticalModel Implementation
// ============================================================================

TacticalModel::TacticalModel() : model_("TacticalModel") {
    // Build architecture matching Python implementation
    // Input: 16 features
    // Dense(128) -> ReLU -> Dropout(0.2)
    // Dense(64) -> ReLU -> Dropout(0.2)
    // Dense(32) -> ReLU
    // Dense(16) -> ReLU
    // Dense(8) -> Softmax

    model_.add<DenseLayer>(16, 128, Activation::ReLU);
    model_.add<DropoutLayer>(0.2f);
    model_.add<DenseLayer>(128, 64, Activation::ReLU);
    model_.add<DropoutLayer>(0.2f);
    model_.add<DenseLayer>(64, 32, Activation::ReLU);
    model_.add<DenseLayer>(32, 16, Activation::ReLU);
    model_.add<DenseLayer>(16, 8, Activation::Softmax);
}

TacticalModel::Action TacticalModel::predict(const TacticalInput& input) {
    Tensor probs = predict_probs(input);

    int best_action = 0;
    float best_prob = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_action = static_cast<int>(i);
        }
    }

    return static_cast<Action>(best_action);
}

Tensor TacticalModel::predict_probs(const TacticalInput& input) {
    Tensor input_tensor = input.to_tensor();
    return model_.predict(input_tensor.reshape(TensorShape({1, 16})));
}

Tensor TacticalModel::forward(const Tensor& input) {
    return model_.forward(input);
}

// ============================================================================
// EchoValueModel Implementation
// ============================================================================

EchoValueModel::EchoValueModel() : model_("EchoValueModel") {
    // Build architecture matching Python implementation
    // Input: 6 features
    // Dense(64) -> ReLU
    // Dense(32) -> ReLU
    // Dense(16) -> ReLU
    // Dense(1) -> Linear (regression output)

    model_.add<DenseLayer>(6, 64, Activation::ReLU);
    model_.add<DenseLayer>(64, 32, Activation::ReLU);
    model_.add<DenseLayer>(32, 16, Activation::ReLU);
    model_.add<DenseLayer>(16, 1, Activation::None);
}

float EchoValueModel::predict(const EchoInput& input) {
    Tensor input_tensor = input.to_tensor();
    Tensor output = model_.predict(input_tensor.reshape(TensorShape({1, 6})));
    return output[0];
}

Tensor EchoValueModel::forward(const Tensor& input) {
    return model_.forward(input);
}

} // namespace ml
} // namespace echosurf
