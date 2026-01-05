/**
 * EchoSurf ML Framework - Tensor Operations Implementation
 *
 * SIMD-optimized tensor operations for high-performance inference.
 * Targets AVX2 for modern CPUs, falls back to SSE4 or scalar.
 */

#include "tensor.h"
#include <cstring>
#include <limits>

namespace echosurf {
namespace ml {

// Tensor in-place operations
void Tensor::add_inplace(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch for add_inplace");
    }

    const size_t n = data_.size();
    float* a = data_.data();
    const float* b = other.data_.data();

#ifdef ECHOSURF_SIMD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_store_ps(a + i, result);
    }
    // Handle remainder
    for (; i < n; ++i) {
        a[i] += b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
#endif
}

void Tensor::mul_inplace(float scalar) {
    const size_t n = data_.size();
    float* a = data_.data();

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 result = _mm256_mul_ps(va, vs);
        _mm256_store_ps(a + i, result);
    }
    for (; i < n; ++i) {
        a[i] *= scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] *= scalar;
    }
#endif
}

void Tensor::add_scalar_inplace(float scalar) {
    const size_t n = data_.size();
    float* a = data_.data();

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 result = _mm256_add_ps(va, vs);
        _mm256_store_ps(a + i, result);
    }
    for (; i < n; ++i) {
        a[i] += scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] += scalar;
    }
#endif
}

namespace ops {

/**
 * Matrix multiplication with SIMD optimization
 *
 * Uses blocking/tiling for cache efficiency.
 * Block sizes tuned for L1/L2 cache.
 */
void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }

    const size_t M = A.shape()[0];
    const size_t K = A.shape()[1];
    const size_t N = B.shape()[1];

    if (B.shape()[0] != K) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }

    // Resize output if needed
    if (C.shape()[0] != M || C.shape()[1] != N) {
        C = Tensor(TensorShape({M, N}));
    }
    C.zero();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    // Block size for cache efficiency (64KB L1 cache typical)
    constexpr size_t BLOCK_SIZE = 32;

#ifdef ECHOSURF_SIMD_AVX2
    // AVX2 optimized matrix multiplication with blocking
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        const size_t i_end = std::min(i0 + BLOCK_SIZE, M);

        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            const size_t k_end = std::min(k0 + BLOCK_SIZE, K);

            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                const size_t j_end = std::min(j0 + BLOCK_SIZE, N);

                // Process block
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        const float a_ik = a[i * K + k];
                        __m256 va = _mm256_set1_ps(a_ik);

                        size_t j = j0;
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&b[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&c[i * N + j]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&c[i * N + j], vc);
                        }
                        // Remainder
                        for (; j < j_end; ++j) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
#else
    // Scalar fallback with blocking for cache efficiency
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                for (size_t i = i0; i < std::min(i0 + BLOCK_SIZE, M); ++i) {
                    for (size_t k = k0; k < std::min(k0 + BLOCK_SIZE, K); ++k) {
                        const float a_ik = a[i * K + k];
                        for (size_t j = j0; j < std::min(j0 + BLOCK_SIZE, N); ++j) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
#endif
}

void batched_matmul(const Tensor& A, const Tensor& B, Tensor& C, size_t batch_size) {
    // For single-batch inference (most common case), delegate to standard matmul
    if (batch_size == 1) {
        matmul(A, B, C);
        return;
    }

    // TODO: Implement true batched matmul for batch inference
    matmul(A, B, C);
}

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Shape mismatch for add");
    }
    Tensor result(a.shape());
    const size_t n = a.size();

#ifdef ECHOSURF_SIMD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data() + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result.data() + i, vr);
    }
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
#endif
    return result;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Shape mismatch for subtract");
    }
    Tensor result(a.shape());
    const size_t n = a.size();

#ifdef ECHOSURF_SIMD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data() + i);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(result.data() + i, vr);
    }
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
#endif
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Shape mismatch for multiply");
    }
    Tensor result(a.shape());
    const size_t n = a.size();

#ifdef ECHOSURF_SIMD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data() + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(result.data() + i, vr);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
#endif
    return result;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Shape mismatch for divide");
    }
    Tensor result(a.shape());
    const size_t n = a.size();

#ifdef ECHOSURF_SIMD_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data() + i);
        __m256 vr = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(result.data() + i, vr);
    }
    for (; i < n; ++i) {
        result[i] = a[i] / b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] / b[i];
    }
#endif
    return result;
}

Tensor add_scalar(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    const size_t n = a.size();

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vr = _mm256_add_ps(va, vs);
        _mm256_storeu_ps(result.data() + i, vr);
    }
    for (; i < n; ++i) {
        result[i] = a[i] + scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + scalar;
    }
#endif
    return result;
}

Tensor mul_scalar(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    const size_t n = a.size();

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(result.data() + i, vr);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] * scalar;
    }
#endif
    return result;
}

// Reductions
float sum(const Tensor& a) {
    const size_t n = a.size();
    float result = 0.0f;

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        vsum = _mm256_add_ps(vsum, va);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    result = _mm_cvtss_f32(sums);

    for (; i < n; ++i) {
        result += a[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result += a[i];
    }
#endif
    return result;
}

float mean(const Tensor& a) {
    return sum(a) / static_cast<float>(a.size());
}

float max(const Tensor& a) {
    if (a.empty()) return std::numeric_limits<float>::lowest();

    const size_t n = a.size();
    float result = a[0];

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vmax = _mm256_set1_ps(a[0]);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        vmax = _mm256_max_ps(vmax, va);
    }
    // Horizontal max
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);
    for (int j = 0; j < 8; ++j) {
        result = std::max(result, tmp[j]);
    }
    for (; i < n; ++i) {
        result = std::max(result, a[i]);
    }
#else
    for (size_t i = 1; i < n; ++i) {
        result = std::max(result, a[i]);
    }
#endif
    return result;
}

float min(const Tensor& a) {
    if (a.empty()) return std::numeric_limits<float>::max();

    const size_t n = a.size();
    float result = a[0];

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vmin = _mm256_set1_ps(a[0]);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        vmin = _mm256_min_ps(vmin, va);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmin);
    for (int j = 0; j < 8; ++j) {
        result = std::min(result, tmp[j]);
    }
    for (; i < n; ++i) {
        result = std::min(result, a[i]);
    }
#else
    for (size_t i = 1; i < n; ++i) {
        result = std::min(result, a[i]);
    }
#endif
    return result;
}

/**
 * ReLU activation: max(0, x)
 * SIMD optimized for fast inference
 */
void relu(const Tensor& input, Tensor& output) {
    const size_t n = input.size();
    if (output.size() != n) {
        output = Tensor(input.shape());
    }

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(input.data() + i);
        __m256 vr = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(output.data() + i, vr);
    }
    for (; i < n; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
#endif
}

void relu_inplace(Tensor& x) {
    const size_t n = x.size();

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(x.data() + i);
        __m256 vr = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(x.data() + i, vr);
    }
    for (; i < n; ++i) {
        x[i] = std::max(0.0f, x[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::max(0.0f, x[i]);
    }
#endif
}

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 * Approximation for speed when exact values not critical
 */
void sigmoid(const Tensor& input, Tensor& output) {
    const size_t n = input.size();
    if (output.size() != n) {
        output = Tensor(input.shape());
    }

    // Fast sigmoid approximation for inference
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        // Clamp to avoid overflow
        x = std::max(-88.0f, std::min(88.0f, x));
        output[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

void sigmoid_inplace(Tensor& x) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        float val = x[i];
        val = std::max(-88.0f, std::min(88.0f, val));
        x[i] = 1.0f / (1.0f + std::exp(-val));
    }
}

/**
 * Tanh activation
 */
void tanh_activation(const Tensor& input, Tensor& output) {
    const size_t n = input.size();
    if (output.size() != n) {
        output = Tensor(input.shape());
    }

    for (size_t i = 0; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void tanh_inplace(Tensor& x) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::tanh(x[i]);
    }
}

/**
 * Softmax activation
 * Numerically stable version using max subtraction
 */
void softmax(const Tensor& input, Tensor& output) {
    const size_t n = input.size();
    if (output.size() != n) {
        output = Tensor(input.shape());
    }

    // Find max for numerical stability
    float max_val = max(input);

    // Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    // Normalize
    const float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

/**
 * Leaky ReLU: max(alpha * x, x)
 */
void leaky_relu(const Tensor& input, Tensor& output, float alpha) {
    const size_t n = input.size();
    if (output.size() != n) {
        output = Tensor(input.shape());
    }

#ifdef ECHOSURF_SIMD_AVX2
    __m256 vzero = _mm256_setzero_ps();
    __m256 valpha = _mm256_set1_ps(alpha);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(input.data() + i);
        __m256 neg = _mm256_mul_ps(va, valpha);
        __m256 mask = _mm256_cmp_ps(va, vzero, _CMP_GT_OQ);
        __m256 vr = _mm256_blendv_ps(neg, va, mask);
        _mm256_storeu_ps(output.data() + i, vr);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
#endif
}

/**
 * Fused bias addition (reduces memory bandwidth)
 */
void add_bias(const Tensor& input, const Tensor& bias, Tensor& output) {
    if (input.ndim() != 2) {
        throw std::invalid_argument("add_bias expects 2D input");
    }

    const size_t batch = input.shape()[0];
    const size_t features = input.shape()[1];

    if (bias.size() != features) {
        throw std::invalid_argument("Bias size must match feature dimension");
    }

    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    for (size_t b = 0; b < batch; ++b) {
        const float* in_row = input.data() + b * features;
        float* out_row = output.data() + b * features;

#ifdef ECHOSURF_SIMD_AVX2
        size_t i = 0;
        for (; i + 8 <= features; i += 8) {
            __m256 vi = _mm256_loadu_ps(in_row + i);
            __m256 vb = _mm256_loadu_ps(bias.data() + i);
            __m256 vr = _mm256_add_ps(vi, vb);
            _mm256_storeu_ps(out_row + i, vr);
        }
        for (; i < features; ++i) {
            out_row[i] = in_row[i] + bias[i];
        }
#else
        for (size_t i = 0; i < features; ++i) {
            out_row[i] = in_row[i] + bias[i];
        }
#endif
    }
}

/**
 * Dropout (inference mode)
 */
void dropout(const Tensor& input, Tensor& output, float keep_prob, bool training) {
    if (output.shape() != input.shape()) {
        output = Tensor(input.shape());
    }

    if (!training) {
        // In inference mode, just copy (no dropout applied)
        std::copy(input.data(), input.data() + input.size(), output.data());
    } else {
        // Training mode: scale by keep_prob
        // Note: For inference, we typically don't drop, just scale
        const float scale = 1.0f / keep_prob;
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i] * scale;
        }
    }
}

} // namespace ops
} // namespace ml
} // namespace echosurf
