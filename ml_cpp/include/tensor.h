/**
 * EchoSurf ML Framework - Tensor Library
 *
 * High-performance tensor operations for neural network inference.
 * Optimized for gaming applications requiring <10ms reflex response.
 *
 * Inspired by sensorimotor coordination tensor concepts:
 * - Covariant/contravariant transformations
 * - Efficient memory access patterns
 * - SIMD vectorization support
 */

#ifndef ECHOSURF_TENSOR_H
#define ECHOSURF_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <functional>

#ifdef __AVX2__
#include <immintrin.h>
#define ECHOSURF_SIMD_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define ECHOSURF_SIMD_SSE4 1
#endif

namespace echosurf {
namespace ml {

// Memory alignment for SIMD operations (32 bytes for AVX)
constexpr size_t MEMORY_ALIGNMENT = 32;
constexpr size_t SIMD_VECTOR_SIZE = 8;  // 8 floats in AVX register

/**
 * Aligned memory allocator for SIMD-optimized operations
 */
template<typename T>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, MEMORY_ALIGNMENT, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
        free(ptr);
    }
};

using AlignedFloatVector = std::vector<float, AlignedAllocator<float>>;

/**
 * Shape descriptor for tensors
 */
class TensorShape {
public:
    TensorShape() : dims_{}, total_size_(0) {}

    explicit TensorShape(std::initializer_list<size_t> dims)
        : dims_(dims) {
        compute_total_size();
    }

    explicit TensorShape(const std::vector<size_t>& dims)
        : dims_(dims) {
        compute_total_size();
    }

    size_t ndim() const { return dims_.size(); }
    size_t size() const { return total_size_; }
    size_t operator[](size_t i) const { return dims_[i]; }

    const std::vector<size_t>& dims() const { return dims_; }

    bool operator==(const TensorShape& other) const {
        return dims_ == other.dims_;
    }

    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        std::string s = "(";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(dims_[i]);
        }
        s += ")";
        return s;
    }

private:
    std::vector<size_t> dims_;
    size_t total_size_;

    void compute_total_size() {
        total_size_ = dims_.empty() ? 0 : 1;
        for (size_t d : dims_) {
            total_size_ *= d;
        }
    }
};

/**
 * High-performance Tensor class
 *
 * Memory layout: Row-major (C-style) for cache efficiency
 * Supports:
 * - SIMD vectorized operations
 * - In-place operations to minimize allocations
 * - Broadcasting for element-wise ops
 */
class Tensor {
public:
    // Constructors
    Tensor() : shape_(), data_() {}

    explicit Tensor(const TensorShape& shape)
        : shape_(shape), data_(shape.size(), 0.0f) {}

    Tensor(const TensorShape& shape, float fill_value)
        : shape_(shape), data_(shape.size(), fill_value) {}

    Tensor(const TensorShape& shape, const float* data)
        : shape_(shape), data_(data, data + shape.size()) {}

    Tensor(const TensorShape& shape, const std::vector<float>& data)
        : shape_(shape), data_(data.begin(), data.end()) {
        if (data.size() != shape.size()) {
            throw std::invalid_argument("Data size doesn't match shape");
        }
    }

    // Factory methods
    static Tensor zeros(const TensorShape& shape) {
        return Tensor(shape, 0.0f);
    }

    static Tensor ones(const TensorShape& shape) {
        return Tensor(shape, 1.0f);
    }

    static Tensor from_vector(const std::vector<float>& data) {
        return Tensor(TensorShape({data.size()}), data);
    }

    // Accessors
    const TensorShape& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    size_t ndim() const { return shape_.ndim(); }
    bool empty() const { return data_.empty(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Element access
    float& operator[](size_t i) { return data_[i]; }
    float operator[](size_t i) const { return data_[i]; }

    float& at(size_t i) {
        if (i >= data_.size()) throw std::out_of_range("Index out of range");
        return data_[i];
    }

    float at(size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("Index out of range");
        return data_[i];
    }

    // 2D access (row-major)
    float& at(size_t row, size_t col) {
        return data_[row * shape_[1] + col];
    }

    float at(size_t row, size_t col) const {
        return data_[row * shape_[1] + col];
    }

    // Reshape (view only, no data copy)
    Tensor reshape(const TensorShape& new_shape) const {
        if (new_shape.size() != shape_.size()) {
            throw std::invalid_argument("Cannot reshape: size mismatch");
        }
        Tensor result(new_shape);
        std::copy(data_.begin(), data_.end(), result.data_.begin());
        return result;
    }

    // In-place operations (minimize allocations for real-time performance)
    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void zero() { fill(0.0f); }

    // SIMD-optimized operations
    void add_inplace(const Tensor& other);
    void mul_inplace(float scalar);
    void add_scalar_inplace(float scalar);

    // Copy operations
    Tensor copy() const {
        Tensor result(shape_);
        std::copy(data_.begin(), data_.end(), result.data_.begin());
        return result;
    }

    // Iterators
    AlignedFloatVector::iterator begin() { return data_.begin(); }
    AlignedFloatVector::iterator end() { return data_.end(); }
    AlignedFloatVector::const_iterator begin() const { return data_.begin(); }
    AlignedFloatVector::const_iterator end() const { return data_.end(); }

private:
    TensorShape shape_;
    AlignedFloatVector data_;
};

/**
 * SIMD-optimized tensor operations
 *
 * These operations leverage AVX2/SSE4 instructions for maximum throughput.
 * Critical for achieving <10ms reflex model inference.
 */
namespace ops {

// Matrix multiplication: C = A @ B
// A: (M, K), B: (K, N) -> C: (M, N)
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Batched matrix multiplication for batch inference
void batched_matmul(const Tensor& A, const Tensor& B, Tensor& C, size_t batch_size);

// Element-wise operations (SIMD accelerated)
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);

// Scalar operations
Tensor add_scalar(const Tensor& a, float scalar);
Tensor mul_scalar(const Tensor& a, float scalar);

// Reductions
float sum(const Tensor& a);
float mean(const Tensor& a);
float max(const Tensor& a);
float min(const Tensor& a);

// Activation functions (SIMD optimized)
void relu(const Tensor& input, Tensor& output);
void relu_inplace(Tensor& x);
void sigmoid(const Tensor& input, Tensor& output);
void sigmoid_inplace(Tensor& x);
void tanh_activation(const Tensor& input, Tensor& output);
void tanh_inplace(Tensor& x);
void softmax(const Tensor& input, Tensor& output);
void leaky_relu(const Tensor& input, Tensor& output, float alpha = 0.01f);

// Bias addition (fused operation for efficiency)
void add_bias(const Tensor& input, const Tensor& bias, Tensor& output);

// Dropout (inference mode: no-op, training: scale)
void dropout(const Tensor& input, Tensor& output, float keep_prob, bool training);

} // namespace ops

/**
 * Memory pool for tensor allocation
 *
 * Pre-allocates memory to avoid malloc overhead during inference.
 * Critical for consistent sub-10ms latency.
 */
class TensorPool {
public:
    explicit TensorPool(size_t initial_capacity = 1024 * 1024)
        : capacity_(initial_capacity) {
        pool_.reserve(16);  // Typical number of intermediate tensors
    }

    Tensor& acquire(const TensorShape& shape) {
        for (auto& tensor : pool_) {
            if (tensor.shape() == shape) {
                tensor.zero();
                return tensor;
            }
        }
        pool_.emplace_back(shape);
        return pool_.back();
    }

    void clear() {
        pool_.clear();
    }

private:
    size_t capacity_;
    std::vector<Tensor> pool_;
};

} // namespace ml
} // namespace echosurf

#endif // ECHOSURF_TENSOR_H
