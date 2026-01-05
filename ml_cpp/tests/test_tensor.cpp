/**
 * EchoSurf ML Framework - Tensor Tests
 */

#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace echosurf::ml;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " << #name << "... "; \
    test_##name(); \
    std::cout << "PASS\n"; \
} while(0)

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))

TEST(tensor_shape) {
    TensorShape shape({2, 3, 4});
    ASSERT_EQ(shape.ndim(), 3);
    ASSERT_EQ(shape.size(), 24);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
}

TEST(tensor_creation) {
    Tensor t1(TensorShape({4, 4}));
    ASSERT_EQ(t1.size(), 16);
    ASSERT_EQ(t1.ndim(), 2);

    Tensor t2(TensorShape({3, 3}), 1.5f);
    for (size_t i = 0; i < t2.size(); ++i) {
        ASSERT_NEAR(t2[i], 1.5f, 1e-6f);
    }

    Tensor zeros = Tensor::zeros(TensorShape({10}));
    for (size_t i = 0; i < zeros.size(); ++i) {
        ASSERT_NEAR(zeros[i], 0.0f, 1e-6f);
    }

    Tensor ones = Tensor::ones(TensorShape({5, 5}));
    for (size_t i = 0; i < ones.size(); ++i) {
        ASSERT_NEAR(ones[i], 1.0f, 1e-6f);
    }
}

TEST(tensor_access) {
    Tensor t(TensorShape({3, 4}));
    for (size_t i = 0; i < 12; ++i) {
        t[i] = static_cast<float>(i);
    }

    ASSERT_NEAR(t.at(0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(t.at(0, 3), 3.0f, 1e-6f);
    ASSERT_NEAR(t.at(2, 3), 11.0f, 1e-6f);
}

TEST(tensor_operations) {
    Tensor a(TensorShape({4}), std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b(TensorShape({4}), std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f});

    // Add
    Tensor c = ops::add(a, b);
    ASSERT_NEAR(c[0], 3.0f, 1e-6f);
    ASSERT_NEAR(c[1], 5.0f, 1e-6f);
    ASSERT_NEAR(c[2], 7.0f, 1e-6f);
    ASSERT_NEAR(c[3], 9.0f, 1e-6f);

    // Multiply
    Tensor d = ops::multiply(a, b);
    ASSERT_NEAR(d[0], 2.0f, 1e-6f);
    ASSERT_NEAR(d[1], 6.0f, 1e-6f);
    ASSERT_NEAR(d[2], 12.0f, 1e-6f);
    ASSERT_NEAR(d[3], 20.0f, 1e-6f);

    // Scalar operations
    Tensor e = ops::mul_scalar(a, 2.0f);
    ASSERT_NEAR(e[0], 2.0f, 1e-6f);
    ASSERT_NEAR(e[3], 8.0f, 1e-6f);
}

TEST(tensor_reductions) {
    Tensor a(TensorShape({4}), std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

    ASSERT_NEAR(ops::sum(a), 10.0f, 1e-6f);
    ASSERT_NEAR(ops::mean(a), 2.5f, 1e-6f);
    ASSERT_NEAR(ops::max(a), 4.0f, 1e-6f);
    ASSERT_NEAR(ops::min(a), 1.0f, 1e-6f);
}

TEST(tensor_matmul) {
    // 2x3 @ 3x2 = 2x2
    Tensor A(TensorShape({2, 3}), std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    Tensor B(TensorShape({3, 2}), std::vector<float>{
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    });

    Tensor C;
    ops::matmul(A, B, C);

    ASSERT_EQ(C.shape()[0], 2);
    ASSERT_EQ(C.shape()[1], 2);

    // Expected: [[58, 64], [139, 154]]
    ASSERT_NEAR(C.at(0, 0), 58.0f, 1e-4f);
    ASSERT_NEAR(C.at(0, 1), 64.0f, 1e-4f);
    ASSERT_NEAR(C.at(1, 0), 139.0f, 1e-4f);
    ASSERT_NEAR(C.at(1, 1), 154.0f, 1e-4f);
}

TEST(tensor_relu) {
    Tensor x(TensorShape({6}), std::vector<float>{
        -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f
    });

    Tensor y;
    ops::relu(x, y);

    ASSERT_NEAR(y[0], 0.0f, 1e-6f);
    ASSERT_NEAR(y[1], 0.0f, 1e-6f);
    ASSERT_NEAR(y[2], 0.0f, 1e-6f);
    ASSERT_NEAR(y[3], 0.5f, 1e-6f);
    ASSERT_NEAR(y[4], 1.0f, 1e-6f);
    ASSERT_NEAR(y[5], 2.0f, 1e-6f);
}

TEST(tensor_sigmoid) {
    Tensor x(TensorShape({3}), std::vector<float>{-10.0f, 0.0f, 10.0f});

    Tensor y;
    ops::sigmoid(x, y);

    ASSERT_NEAR(y[0], 0.0f, 0.001f);  // sigmoid(-10) ~ 0
    ASSERT_NEAR(y[1], 0.5f, 1e-6f);   // sigmoid(0) = 0.5
    ASSERT_NEAR(y[2], 1.0f, 0.001f);  // sigmoid(10) ~ 1
}

TEST(tensor_softmax) {
    Tensor x(TensorShape({4}), std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

    Tensor y;
    ops::softmax(x, y);

    // Sum should be 1
    float sum = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        sum += y[i];
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    // Larger input -> larger probability
    assert(y[3] > y[2]);
    assert(y[2] > y[1]);
    assert(y[1] > y[0]);
}

TEST(tensor_inplace) {
    Tensor a(TensorShape({4}), std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b(TensorShape({4}), std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});

    a.add_inplace(b);
    ASSERT_NEAR(a[0], 2.0f, 1e-6f);
    ASSERT_NEAR(a[3], 5.0f, 1e-6f);

    a.mul_inplace(2.0f);
    ASSERT_NEAR(a[0], 4.0f, 1e-6f);
    ASSERT_NEAR(a[3], 10.0f, 1e-6f);
}

int main() {
    std::cout << "=== EchoSurf ML Tensor Tests ===\n\n";

    RUN_TEST(tensor_shape);
    RUN_TEST(tensor_creation);
    RUN_TEST(tensor_access);
    RUN_TEST(tensor_operations);
    RUN_TEST(tensor_reductions);
    RUN_TEST(tensor_matmul);
    RUN_TEST(tensor_relu);
    RUN_TEST(tensor_sigmoid);
    RUN_TEST(tensor_softmax);
    RUN_TEST(tensor_inplace);

    std::cout << "\nAll tensor tests passed!\n";
    return 0;
}
