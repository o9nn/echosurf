/**
 * EchoSurf ML Framework - Model and Inference Engine
 *
 * High-performance inference engine for gaming applications.
 *
 * Performance targets (from learning loop paradigm):
 * - Reflex model: <10ms inference (lightning response)
 * - Tactical model: <50ms inference (strategic planning)
 *
 * Control through Learning Loops:
 * - Idea: Model receives sensory input
 * - Test: Forward pass computes prediction
 * - Learn: Results inform future behavior
 * - Outcome: Action is taken
 */

#ifndef ECHOSURF_MODEL_H
#define ECHOSURF_MODEL_H

#include "tensor.h"
#include "layers.h"
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <functional>

namespace echosurf {
namespace ml {

/**
 * Performance metrics for real-time monitoring
 */
struct InferenceMetrics {
    double last_inference_ms;
    double avg_inference_ms;
    double min_inference_ms;
    double max_inference_ms;
    size_t total_inferences;

    InferenceMetrics()
        : last_inference_ms(0), avg_inference_ms(0)
        , min_inference_ms(std::numeric_limits<double>::max())
        , max_inference_ms(0), total_inferences(0) {}

    void update(double time_ms) {
        last_inference_ms = time_ms;
        min_inference_ms = std::min(min_inference_ms, time_ms);
        max_inference_ms = std::max(max_inference_ms, time_ms);

        // Running average
        total_inferences++;
        avg_inference_ms = avg_inference_ms +
                          (time_ms - avg_inference_ms) / total_inferences;
    }

    void reset() {
        *this = InferenceMetrics();
    }
};

/**
 * Sequential Neural Network Model
 *
 * Implements a feedforward neural network as a sequence of layers.
 * Optimized for low-latency inference.
 */
class SequentialModel {
public:
    SequentialModel() = default;
    explicit SequentialModel(const std::string& name) : name_(name) {}

    // Layer management
    void add(std::unique_ptr<Layer> layer);

    template<typename LayerType, typename... Args>
    void add(Args&&... args) {
        add(std::make_unique<LayerType>(std::forward<Args>(args)...));
    }

    // Forward pass
    Tensor forward(const Tensor& input);

    // Inference with timing
    Tensor predict(const Tensor& input);

    // Batch inference
    Tensor predict_batch(const std::vector<Tensor>& inputs);

    // Model information
    const std::string& name() const { return name_; }
    size_t num_layers() const { return layers_.size(); }
    size_t total_params() const;

    // Layer access
    Layer& layer(size_t index) { return *layers_[index]; }
    const Layer& layer(size_t index) const { return *layers_[index]; }

    // Performance metrics
    const InferenceMetrics& metrics() const { return metrics_; }
    void reset_metrics() { metrics_.reset(); }

    // Output shape computation
    TensorShape output_shape(const TensorShape& input_shape) const;

    // Summary
    std::string summary() const;

private:
    std::string name_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<Tensor> intermediate_outputs_;
    InferenceMetrics metrics_;
};

/**
 * Reflex Model - Optimized for <10ms response
 *
 * Architecture matching Python implementation:
 * Input(8) -> Dense(128, ReLU) -> Dropout(0.1) ->
 * Dense(64, ReLU) -> Dropout(0.1) ->
 * Dense(32, ReLU) -> Dense(4, Softmax)
 */
class ReflexModel {
public:
    // Reflex input features
    struct ReflexInput {
        float threat_proximity;    // 0-1: how close is threat
        float threat_direction;    // -1 to 1: direction
        float player_state;        // 0-1: health/shield
        float movement_momentum;   // current momentum
        float time_pressure;       // urgency factor
        float cover_availability;  // nearby cover
        float aim_confidence;      // aim lock quality
        float situation_clarity;   // visibility/info

        Tensor to_tensor() const {
            return Tensor::from_vector({
                threat_proximity, threat_direction, player_state,
                movement_momentum, time_pressure, cover_availability,
                aim_confidence, situation_clarity
            });
        }
    };

    // Reflex output actions
    enum class Action : int {
        Dodge = 0,
        CounterAttack = 1,
        TakeCover = 2,
        HoldPosition = 3
    };

    ReflexModel();

    // Predict action from input
    Action predict(const ReflexInput& input);

    // Get action probabilities
    Tensor predict_probs(const ReflexInput& input);

    // Raw tensor inference
    Tensor forward(const Tensor& input);

    // Model access
    SequentialModel& model() { return model_; }
    const SequentialModel& model() const { return model_; }

    // Performance check
    bool meets_latency_target() const {
        return model_.metrics().avg_inference_ms < LATENCY_TARGET_MS;
    }

    static constexpr double LATENCY_TARGET_MS = 10.0;

private:
    SequentialModel model_;
};

/**
 * Tactical Model - Strategic decision making
 *
 * Architecture matching Python implementation:
 * Input(16) -> Dense(128, ReLU) -> Dropout(0.2) ->
 * Dense(64, ReLU) -> Dropout(0.2) ->
 * Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(8, Softmax)
 */
class TacticalModel {
public:
    // Tactical input features
    struct TacticalInput {
        float threat_level;        // overall threat
        float health;              // current health
        float ammo;                // ammo percentage
        float armor;               // armor level
        float items;               // item count
        float currency;            // resource level
        float pos_x, pos_y, pos_z; // position
        float ally_strength;       // team strength
        float ally_distance;       // distance to allies
        float ally_health;         // ally health
        float enemy_strength;      // enemy strength
        float enemy_distance;      // enemy distance
        float enemy_count;         // number of enemies
        float objective_distance;  // distance to objective

        Tensor to_tensor() const {
            return Tensor::from_vector({
                threat_level, health, ammo, armor, items, currency,
                pos_x, pos_y, pos_z,
                ally_strength, ally_distance, ally_health,
                enemy_strength, enemy_distance, enemy_count,
                objective_distance
            });
        }
    };

    // Tactical actions
    enum class Action : int {
        Attack = 0,
        Defend = 1,
        Flank = 2,
        Retreat = 3,
        Heal = 4,
        Resupply = 5,
        Support = 6,
        Objective = 7
    };

    TacticalModel();

    // Predict tactical action
    Action predict(const TacticalInput& input);

    // Get action probabilities
    Tensor predict_probs(const TacticalInput& input);

    // Raw tensor inference
    Tensor forward(const Tensor& input);

    // Model access
    SequentialModel& model() { return model_; }
    const SequentialModel& model() const { return model_; }

    // Performance check
    bool meets_latency_target() const {
        return model_.metrics().avg_inference_ms < LATENCY_TARGET_MS;
    }

    static constexpr double LATENCY_TARGET_MS = 50.0;

private:
    SequentialModel model_;
};

/**
 * Echo Value Model - Content importance prediction
 *
 * Predicts the "echo value" of content in the deep tree echo system.
 */
class EchoValueModel {
public:
    struct EchoInput {
        float content_length;      // normalized length
        float complexity;          // character complexity
        float depth;               // tree depth
        float child_count;         // number of children
        float sibling_count;       // number of siblings
        float historical_value;    // previous echo values

        Tensor to_tensor() const {
            return Tensor::from_vector({
                content_length, complexity, depth,
                child_count, sibling_count, historical_value
            });
        }
    };

    EchoValueModel();

    float predict(const EchoInput& input);
    Tensor forward(const Tensor& input);

    SequentialModel& model() { return model_; }

private:
    SequentialModel model_;
};

/**
 * Model Ensemble - Combines multiple models for robust predictions
 *
 * Implements voting/averaging strategies for improved reliability.
 */
template<typename ModelType, typename InputType>
class ModelEnsemble {
public:
    using PredictFunc = std::function<Tensor(ModelType&, const InputType&)>;

    void add_model(std::unique_ptr<ModelType> model) {
        models_.push_back(std::move(model));
    }

    // Majority voting for classification
    int predict_vote(const InputType& input, PredictFunc predict_fn) {
        std::vector<int> votes(16, 0);  // Max 16 classes

        for (auto& model : models_) {
            Tensor probs = predict_fn(*model, input);
            int best_class = 0;
            float best_prob = probs[0];
            for (size_t i = 1; i < probs.size(); ++i) {
                if (probs[i] > best_prob) {
                    best_prob = probs[i];
                    best_class = static_cast<int>(i);
                }
            }
            votes[best_class]++;
        }

        return std::distance(votes.begin(),
                            std::max_element(votes.begin(), votes.end()));
    }

    // Average probabilities
    Tensor predict_avg(const InputType& input, PredictFunc predict_fn) {
        if (models_.empty()) {
            throw std::runtime_error("Ensemble has no models");
        }

        Tensor avg = predict_fn(*models_[0], input);

        for (size_t i = 1; i < models_.size(); ++i) {
            Tensor probs = predict_fn(*models_[i], input);
            avg.add_inplace(probs);
        }

        avg.mul_inplace(1.0f / models_.size());
        return avg;
    }

private:
    std::vector<std::unique_ptr<ModelType>> models_;
};

} // namespace ml
} // namespace echosurf

#endif // ECHOSURF_MODEL_H
