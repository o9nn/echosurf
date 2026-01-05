/**
 * EchoSurf ML Framework - Inference Benchmarks
 *
 * Measures inference latency for real-time gaming applications.
 *
 * Performance Targets:
 * - Reflex Model: <10ms (lightning response)
 * - Tactical Model: <50ms (strategic planning)
 * - Echo Value Model: <5ms (content evaluation)
 */

#include "tensor.h"
#include "layers.h"
#include "model.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

using namespace echosurf::ml;

// Timer utility
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Statistics helper
struct Statistics {
    double mean;
    double min;
    double max;
    double std_dev;
    double p50;  // median
    double p95;
    double p99;

    static Statistics compute(std::vector<double>& times) {
        Statistics stats;

        if (times.empty()) {
            return stats;
        }

        std::sort(times.begin(), times.end());

        stats.min = times.front();
        stats.max = times.back();
        stats.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        size_t n = times.size();
        stats.p50 = times[n / 2];
        stats.p95 = times[static_cast<size_t>(n * 0.95)];
        stats.p99 = times[static_cast<size_t>(n * 0.99)];

        double sq_sum = 0.0;
        for (double t : times) {
            sq_sum += (t - stats.mean) * (t - stats.mean);
        }
        stats.std_dev = std::sqrt(sq_sum / n);

        return stats;
    }

    void print(const std::string& name, double target_ms) const {
        std::cout << "\n" << name << " Results:\n";
        std::cout << std::string(50, '-') << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Mean:    " << std::setw(10) << mean << " ms\n";
        std::cout << "  Min:     " << std::setw(10) << min << " ms\n";
        std::cout << "  Max:     " << std::setw(10) << max << " ms\n";
        std::cout << "  StdDev:  " << std::setw(10) << std_dev << " ms\n";
        std::cout << "  P50:     " << std::setw(10) << p50 << " ms\n";
        std::cout << "  P95:     " << std::setw(10) << p95 << " ms\n";
        std::cout << "  P99:     " << std::setw(10) << p99 << " ms\n";
        std::cout << std::string(50, '-') << "\n";
        std::cout << "  Target:  " << std::setw(10) << target_ms << " ms\n";
        std::cout << "  Status:  " << (p95 < target_ms ? "PASS" : "FAIL") << "\n";
    }
};

// Generate random input
Tensor random_input(const TensorShape& shape, std::mt19937& rng) {
    Tensor t(shape);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = dist(rng);
    }
    return t;
}

// Warmup function
void warmup(ReflexModel& model, int iterations = 100) {
    std::mt19937 rng(42);
    for (int i = 0; i < iterations; ++i) {
        ReflexModel::ReflexInput input{
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 200 - 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f
        };
        model.predict(input);
    }
}

void warmup(TacticalModel& model, int iterations = 100) {
    std::mt19937 rng(42);
    for (int i = 0; i < iterations; ++i) {
        TacticalModel::TacticalInput input{
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 200 - 100) / 10.0f,
            static_cast<float>(rng() % 200 - 100) / 10.0f,
            static_cast<float>(rng() % 100) / 10.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 10.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 10.0f,
            static_cast<float>(rng() % 10),
            static_cast<float>(rng() % 100) / 10.0f
        };
        model.predict(input);
    }
}

// Benchmark functions
void benchmark_reflex_model(int iterations = 10000) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "REFLEX MODEL BENCHMARK\n";
    std::cout << "Architecture: Input(8) -> Dense(128,ReLU) -> Dropout ->\n";
    std::cout << "              Dense(64,ReLU) -> Dropout ->\n";
    std::cout << "              Dense(32,ReLU) -> Dense(4,Softmax)\n";
    std::cout << std::string(60, '=') << "\n";

    ReflexModel model;

    std::cout << "Model params: " << model.model().total_params() << "\n";
    std::cout << "Warmup iterations: 100\n";
    std::cout << "Benchmark iterations: " << iterations << "\n";

    // Warmup
    warmup(model);
    model.model().reset_metrics();

    // Benchmark
    std::mt19937 rng(12345);
    std::vector<double> times;
    times.reserve(iterations);

    Timer timer;
    for (int i = 0; i < iterations; ++i) {
        ReflexModel::ReflexInput input{
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 200 - 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f
        };

        timer.start();
        auto action = model.predict(input);
        double elapsed = timer.elapsed_ms();
        times.push_back(elapsed);
        (void)action;  // Prevent optimization
    }

    Statistics stats = Statistics::compute(times);
    stats.print("Reflex Model Inference", ReflexModel::LATENCY_TARGET_MS);

    std::cout << "\nMeets latency target: "
              << (model.meets_latency_target() ? "YES" : "NO") << "\n";
}

void benchmark_tactical_model(int iterations = 10000) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TACTICAL MODEL BENCHMARK\n";
    std::cout << "Architecture: Input(16) -> Dense(128,ReLU) -> Dropout ->\n";
    std::cout << "              Dense(64,ReLU) -> Dropout ->\n";
    std::cout << "              Dense(32,ReLU) -> Dense(16,ReLU) ->\n";
    std::cout << "              Dense(8,Softmax)\n";
    std::cout << std::string(60, '=') << "\n";

    TacticalModel model;

    std::cout << "Model params: " << model.model().total_params() << "\n";
    std::cout << "Warmup iterations: 100\n";
    std::cout << "Benchmark iterations: " << iterations << "\n";

    // Warmup
    warmup(model);
    model.model().reset_metrics();

    // Benchmark
    std::mt19937 rng(12345);
    std::vector<double> times;
    times.reserve(iterations);

    Timer timer;
    for (int i = 0; i < iterations; ++i) {
        TacticalModel::TacticalInput input{
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 200 - 100) / 10.0f,
            static_cast<float>(rng() % 200 - 100) / 10.0f,
            static_cast<float>(rng() % 100) / 10.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 10.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 100.0f,
            static_cast<float>(rng() % 100) / 10.0f,
            static_cast<float>(rng() % 10),
            static_cast<float>(rng() % 100) / 10.0f
        };

        timer.start();
        auto action = model.predict(input);
        double elapsed = timer.elapsed_ms();
        times.push_back(elapsed);
        (void)action;
    }

    Statistics stats = Statistics::compute(times);
    stats.print("Tactical Model Inference", TacticalModel::LATENCY_TARGET_MS);

    std::cout << "\nMeets latency target: "
              << (model.meets_latency_target() ? "YES" : "NO") << "\n";
}

void benchmark_tensor_operations(int iterations = 100000) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TENSOR OPERATIONS BENCHMARK\n";
    std::cout << std::string(60, '=') << "\n";

    std::mt19937 rng(42);

    // Matrix multiplication benchmark
    {
        Tensor A(TensorShape({64, 128}));
        Tensor B(TensorShape({128, 64}));
        Tensor C;

        // Initialize with random values
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < A.size(); ++i) A[i] = dist(rng);
        for (size_t i = 0; i < B.size(); ++i) B[i] = dist(rng);

        // Warmup
        for (int i = 0; i < 100; ++i) {
            ops::matmul(A, B, C);
        }

        // Benchmark
        std::vector<double> times;
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            ops::matmul(A, B, C);
            times.push_back(timer.elapsed_ms());
        }

        Statistics stats = Statistics::compute(times);
        std::cout << "\nMatMul [64x128] @ [128x64]:\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(6)
                  << stats.mean << " ms  (" << (64*128*64*2) / (stats.mean * 1e6)
                  << " GFLOPS)\n";
        std::cout << "  P99:  " << stats.p99 << " ms\n";
    }

    // ReLU benchmark
    {
        Tensor x(TensorShape({1, 1024}));
        Tensor y;

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < x.size(); ++i) x[i] = dist(rng);

        // Warmup
        for (int i = 0; i < 1000; ++i) {
            ops::relu(x, y);
        }

        // Benchmark
        std::vector<double> times;
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            ops::relu(x, y);
            times.push_back(timer.elapsed_ms());
        }

        Statistics stats = Statistics::compute(times);
        std::cout << "\nReLU [1024]:\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(6)
                  << stats.mean * 1000 << " us\n";
        std::cout << "  P99:  " << stats.p99 * 1000 << " us\n";
    }

    // Softmax benchmark
    {
        Tensor x(TensorShape({8}));
        Tensor y;

        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (size_t i = 0; i < x.size(); ++i) x[i] = dist(rng);

        // Warmup
        for (int i = 0; i < 1000; ++i) {
            ops::softmax(x, y);
        }

        // Benchmark
        std::vector<double> times;
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            ops::softmax(x, y);
            times.push_back(timer.elapsed_ms());
        }

        Statistics stats = Statistics::compute(times);
        std::cout << "\nSoftmax [8]:\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(6)
                  << stats.mean * 1000 << " us\n";
        std::cout << "  P99:  " << stats.p99 * 1000 << " us\n";
    }
}

void benchmark_dense_layer(int iterations = 10000) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "DENSE LAYER BENCHMARK\n";
    std::cout << std::string(60, '=') << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Dense 128 -> 64 with ReLU
    {
        DenseLayer layer(128, 64, Activation::ReLU);

        // Initialize weights
        Tensor kernel(TensorShape({128, 64}));
        Tensor bias(TensorShape({64}));
        for (size_t i = 0; i < kernel.size(); ++i) kernel[i] = dist(rng) * 0.1f;
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = dist(rng) * 0.01f;
        layer.set_kernel(kernel);
        layer.set_bias(bias);

        Tensor input(TensorShape({1, 128}));
        for (size_t i = 0; i < input.size(); ++i) input[i] = dist(rng);

        Tensor output;

        // Warmup
        for (int i = 0; i < 100; ++i) {
            layer.forward(input, output);
        }

        // Benchmark
        std::vector<double> times;
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            layer.forward(input, output);
            times.push_back(timer.elapsed_ms());
        }

        Statistics stats = Statistics::compute(times);
        std::cout << "\nDense [128 -> 64, ReLU]:\n";
        std::cout << "  Params: " << layer.param_count() << "\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(6)
                  << stats.mean * 1000 << " us\n";
        std::cout << "  P99:  " << stats.p99 * 1000 << " us\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║       EchoSurf ML Framework - Inference Benchmark        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
#ifdef ECHOSURF_SIMD_AVX2
    std::cout << "║  SIMD: AVX2 enabled                                      ║\n";
#elif defined(ECHOSURF_SIMD_SSE4)
    std::cout << "║  SIMD: SSE4.1 enabled                                    ║\n";
#else
    std::cout << "║  SIMD: Disabled (scalar mode)                            ║\n";
#endif
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    int iterations = 10000;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
    }

    benchmark_tensor_operations(iterations);
    benchmark_dense_layer(iterations);
    benchmark_reflex_model(iterations);
    benchmark_tactical_model(iterations);

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "BENCHMARK COMPLETE\n";
    std::cout << std::string(60, '=') << "\n\n";

    return 0;
}
