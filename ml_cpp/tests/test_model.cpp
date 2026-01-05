/**
 * EchoSurf ML Framework - Model Tests
 */

#include "model.h"
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
#define ASSERT_TRUE(x) assert(x)
#define ASSERT_FALSE(x) assert(!(x))

TEST(sequential_model_creation) {
    SequentialModel model("test_model");

    model.add<DenseLayer>(10, 5, Activation::ReLU);
    model.add<DropoutLayer>(0.1f);
    model.add<DenseLayer>(5, 3, Activation::Softmax);

    ASSERT_EQ(model.num_layers(), 3);
    ASSERT_EQ(model.name(), "test_model");

    // Parameters: 10*5 + 5 + 0 + 5*3 + 3 = 73
    ASSERT_EQ(model.total_params(), 73);
}

TEST(sequential_model_forward) {
    SequentialModel model;

    model.add<DenseLayer>(2, 2, Activation::None);

    // Get the layer and set weights to identity
    auto& layer = dynamic_cast<DenseLayer&>(model.layer(0));

    Tensor kernel(TensorShape({2, 2}));
    kernel[0] = 1.0f; kernel[1] = 0.0f;
    kernel[2] = 0.0f; kernel[3] = 1.0f;

    Tensor bias(TensorShape({2}));
    bias[0] = 0.5f;
    bias[1] = -0.5f;

    layer.set_kernel(kernel);
    layer.set_bias(bias);

    Tensor input(TensorShape({1, 2}), std::vector<float>{1.0f, 2.0f});
    Tensor output = model.forward(input);

    ASSERT_NEAR(output[0], 1.5f, 1e-5f);
    ASSERT_NEAR(output[1], 1.5f, 1e-5f);
}

TEST(sequential_model_predict_timing) {
    SequentialModel model;

    model.add<DenseLayer>(8, 16, Activation::ReLU);
    model.add<DenseLayer>(16, 8, Activation::ReLU);
    model.add<DenseLayer>(8, 4, Activation::Softmax);

    Tensor input(TensorShape({1, 8}));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < 8; ++i) {
        input[i] = dist(rng);
    }

    // Run multiple predictions
    for (int i = 0; i < 100; ++i) {
        model.predict(input);
    }

    const auto& metrics = model.metrics();
    ASSERT_EQ(metrics.total_inferences, 100);
    ASSERT_TRUE(metrics.avg_inference_ms > 0);
    ASSERT_TRUE(metrics.min_inference_ms <= metrics.avg_inference_ms);
    ASSERT_TRUE(metrics.max_inference_ms >= metrics.avg_inference_ms);
}

TEST(reflex_model_architecture) {
    ReflexModel model;

    // Check architecture
    ASSERT_EQ(model.model().num_layers(), 6);  // 4 dense + 2 dropout

    // Total params should be reasonable
    size_t params = model.model().total_params();
    ASSERT_TRUE(params > 1000);
    ASSERT_TRUE(params < 50000);
}

TEST(reflex_model_predict) {
    ReflexModel model;

    ReflexModel::ReflexInput input{
        0.8f,   // threat_proximity
        0.5f,   // threat_direction
        0.9f,   // player_state
        0.3f,   // movement_momentum
        0.7f,   // time_pressure
        0.2f,   // cover_availability
        0.6f,   // aim_confidence
        0.8f    // situation_clarity
    };

    auto action = model.predict(input);

    // Action should be one of the valid values
    ASSERT_TRUE(static_cast<int>(action) >= 0);
    ASSERT_TRUE(static_cast<int>(action) <= 3);
}

TEST(reflex_model_probs) {
    ReflexModel model;

    ReflexModel::ReflexInput input{
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f
    };

    Tensor probs = model.predict_probs(input);

    // Probabilities should sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        sum += probs[i];
        ASSERT_TRUE(probs[i] >= 0.0f);
        ASSERT_TRUE(probs[i] <= 1.0f);
    }
    ASSERT_NEAR(sum, 1.0f, 1e-4f);
}

TEST(tactical_model_architecture) {
    TacticalModel model;

    // Check architecture
    ASSERT_EQ(model.model().num_layers(), 7);  // 5 dense + 2 dropout

    // Total params should be reasonable
    size_t params = model.model().total_params();
    ASSERT_TRUE(params > 5000);
    ASSERT_TRUE(params < 100000);
}

TEST(tactical_model_predict) {
    TacticalModel model;

    TacticalModel::TacticalInput input{
        0.6f,   // threat_level
        0.8f,   // health
        0.5f,   // ammo
        0.7f,   // armor
        3.0f,   // items
        100.0f, // currency
        10.0f,  // pos_x
        20.0f,  // pos_y
        5.0f,   // pos_z
        0.5f,   // ally_strength
        15.0f,  // ally_distance
        0.7f,   // ally_health
        0.4f,   // enemy_strength
        25.0f,  // enemy_distance
        3.0f,   // enemy_count
        50.0f   // objective_distance
    };

    auto action = model.predict(input);

    // Action should be valid
    ASSERT_TRUE(static_cast<int>(action) >= 0);
    ASSERT_TRUE(static_cast<int>(action) <= 7);
}

TEST(tactical_model_probs) {
    TacticalModel model;

    TacticalModel::TacticalInput input{
        0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 50.0f,
        0.0f, 0.0f, 0.0f,
        0.5f, 10.0f, 0.5f,
        0.5f, 20.0f, 2.0f, 30.0f
    };

    Tensor probs = model.predict_probs(input);

    // Probabilities should sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        sum += probs[i];
        ASSERT_TRUE(probs[i] >= 0.0f);
        ASSERT_TRUE(probs[i] <= 1.0f);
    }
    ASSERT_NEAR(sum, 1.0f, 1e-4f);
}

TEST(echo_value_model) {
    EchoValueModel model;

    EchoValueModel::EchoInput input{
        0.5f,   // content_length
        0.3f,   // complexity
        0.2f,   // depth
        3.0f,   // child_count
        2.0f,   // sibling_count
        0.7f    // historical_value
    };

    float value = model.predict(input);

    // Value should be a reasonable float
    ASSERT_TRUE(std::isfinite(value));
}

TEST(inference_metrics) {
    InferenceMetrics metrics;

    ASSERT_EQ(metrics.total_inferences, 0);

    metrics.update(5.0);
    ASSERT_EQ(metrics.total_inferences, 1);
    ASSERT_NEAR(metrics.avg_inference_ms, 5.0, 1e-6);
    ASSERT_NEAR(metrics.min_inference_ms, 5.0, 1e-6);
    ASSERT_NEAR(metrics.max_inference_ms, 5.0, 1e-6);

    metrics.update(10.0);
    ASSERT_EQ(metrics.total_inferences, 2);
    ASSERT_NEAR(metrics.avg_inference_ms, 7.5, 1e-6);
    ASSERT_NEAR(metrics.min_inference_ms, 5.0, 1e-6);
    ASSERT_NEAR(metrics.max_inference_ms, 10.0, 1e-6);

    metrics.reset();
    ASSERT_EQ(metrics.total_inferences, 0);
}

TEST(model_summary) {
    SequentialModel model("TestModel");

    model.add<DenseLayer>(10, 5, Activation::ReLU);
    model.add<DenseLayer>(5, 2, Activation::Softmax);

    std::string summary = model.summary();

    ASSERT_TRUE(summary.find("TestModel") != std::string::npos);
    ASSERT_TRUE(summary.find("dense") != std::string::npos);
}

TEST(model_latency_target) {
    ReflexModel reflex;
    TacticalModel tactical;

    // Initially, latency targets should be undefined (no inferences)
    // After running some predictions, check the targets

    ReflexModel::ReflexInput reflex_input{
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f
    };

    TacticalModel::TacticalInput tactical_input{
        0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 50.0f,
        0.0f, 0.0f, 0.0f,
        0.5f, 10.0f, 0.5f,
        0.5f, 20.0f, 2.0f, 30.0f
    };

    // Run warmup
    for (int i = 0; i < 100; ++i) {
        reflex.predict(reflex_input);
        tactical.predict(tactical_input);
    }

    // Check metrics exist
    ASSERT_TRUE(reflex.model().metrics().total_inferences == 100);
    ASSERT_TRUE(tactical.model().metrics().total_inferences == 100);

    // Latency targets
    ASSERT_NEAR(ReflexModel::LATENCY_TARGET_MS, 10.0, 1e-6);
    ASSERT_NEAR(TacticalModel::LATENCY_TARGET_MS, 50.0, 1e-6);
}

int main() {
    std::cout << "=== EchoSurf ML Model Tests ===\n\n";

    RUN_TEST(sequential_model_creation);
    RUN_TEST(sequential_model_forward);
    RUN_TEST(sequential_model_predict_timing);
    RUN_TEST(reflex_model_architecture);
    RUN_TEST(reflex_model_predict);
    RUN_TEST(reflex_model_probs);
    RUN_TEST(tactical_model_architecture);
    RUN_TEST(tactical_model_predict);
    RUN_TEST(tactical_model_probs);
    RUN_TEST(echo_value_model);
    RUN_TEST(inference_metrics);
    RUN_TEST(model_summary);
    RUN_TEST(model_latency_target);

    std::cout << "\nAll model tests passed!\n";
    return 0;
}
