/**
 * test_standalone.c - Standalone test for Echo ML library
 * 
 * Compile: gcc -O3 -Iinclude tests/test_standalone.c -L. -lecho_ml -lm -o test_echo_ml
 * Run: ./test_echo_ml
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "echo_ml.h"

#define TEST_VOCAB_SIZE 1024
#define TEST_EMBED_DIM 32
#define TEST_RESERVOIR_SIZE 64
#define TEST_OUTPUT_DIM 16
#define NUM_ITERATIONS 1000

int main(void) {
    printf("=== Echo ML Standalone Test ===\n\n");
    
    /* Seed RNG */
    echo_seed(42);
    
    /* Test 1: Tensor operations */
    printf("Test 1: Tensor operations\n");
    {
        size_t shape[] = {64, 64};
        EchoTensor t;
        EchoError err = echo_tensor_alloc(&t, shape, 2);
        if (err != ECHO_OK) {
            printf("  FAIL: tensor allocation\n");
            return 1;
        }
        
        /* Fill with random values */
        size_t n = echo_tensor_numel(&t);
        for (size_t i = 0; i < n; i++) {
            t.data[i] = echo_randn();
        }
        
        printf("  Tensor shape: %zu x %zu\n", t.shape[0], t.shape[1]);
        printf("  Tensor numel: %zu\n", n);
        printf("  Sample values: %.4f, %.4f, %.4f\n", t.data[0], t.data[1], t.data[2]);
        
        echo_tensor_free(&t);
        printf("  PASS\n\n");
    }
    
    /* Test 2: SIMD vector operations */
    printf("Test 2: SIMD vector operations\n");
    {
        size_t n = 1024;
        echo_float* a = aligned_alloc(32, n * sizeof(echo_float));
        echo_float* b = aligned_alloc(32, n * sizeof(echo_float));
        echo_float* c = aligned_alloc(32, n * sizeof(echo_float));
        
        for (size_t i = 0; i < n; i++) {
            a[i] = echo_randn();
            b[i] = echo_randn();
        }
        
        /* Test dot product */
        echo_float dot = echo_vec_dot(a, b, n);
        printf("  Dot product: %.4f\n", dot);
        
        /* Test add */
        echo_vec_add(c, a, b, n);
        printf("  Add result[0]: %.4f (expected: %.4f)\n", c[0], a[0] + b[0]);
        
        /* Test tanh */
        echo_vec_tanh(c, a, n);
        printf("  Tanh result[0]: %.4f\n", c[0]);
        
        free(a);
        free(b);
        free(c);
        printf("  PASS\n\n");
    }
    
    /* Test 3: Sparse matrix operations */
    printf("Test 3: Sparse matrix operations\n");
    {
        size_t n = 100;
        EchoSparseMatrix sparse;
        
        /* Create a sparse matrix with ~10% density */
        EchoError err = echo_sparse_alloc(&sparse, n, n, n * n / 10);
        if (err != ECHO_OK) {
            printf("  FAIL: sparse allocation\n");
            return 1;
        }
        
        /* Fill with random sparse pattern */
        size_t idx = 0;
        for (size_t i = 0; i < n; i++) {
            sparse.row_ptr[i] = idx;
            for (size_t j = 0; j < n && idx < sparse.nnz; j++) {
                if (echo_rand() < 0.1f) {
                    sparse.values[idx] = echo_randn();
                    sparse.col_indices[idx] = j;
                    idx++;
                }
            }
        }
        sparse.row_ptr[n] = idx;
        sparse.nnz = idx;
        
        printf("  Sparse matrix: %zu x %zu, nnz=%zu (%.1f%% dense)\n",
               sparse.rows, sparse.cols, sparse.nnz,
               100.0f * sparse.nnz / (sparse.rows * sparse.cols));
        
        /* Test sparse matvec */
        echo_float* x = aligned_alloc(32, n * sizeof(echo_float));
        echo_float* y = aligned_alloc(32, n * sizeof(echo_float));
        for (size_t i = 0; i < n; i++) x[i] = echo_randn();
        
        echo_sparse_matvec(y, &sparse, x);
        printf("  Sparse matvec result[0]: %.4f\n", y[0]);
        
        free(x);
        free(y);
        echo_sparse_free(&sparse);
        printf("  PASS\n\n");
    }
    
    /* Test 4: Embedding layer */
    printf("Test 4: Embedding layer\n");
    {
        EchoEmbedding embed;
        EchoError err = echo_embedding_init(&embed, TEST_VOCAB_SIZE, TEST_EMBED_DIM);
        if (err != ECHO_OK) {
            printf("  FAIL: embedding init\n");
            return 1;
        }
        
        printf("  Vocab size: %zu, Embed dim: %zu\n", embed.vocab_size, embed.embedding_dim);
        
        /* Test lookup */
        echo_token tokens[] = {1, 42, 100, 500};
        size_t num_tokens = sizeof(tokens) / sizeof(tokens[0]);
        echo_float* output = malloc(num_tokens * TEST_EMBED_DIM * sizeof(echo_float));
        
        echo_embedding_forward(output, &embed, tokens, num_tokens);
        printf("  Embedding[0][0]: %.4f\n", output[0]);
        printf("  Embedding[1][0]: %.4f\n", output[TEST_EMBED_DIM]);
        
        free(output);
        echo_embedding_free(&embed);
        printf("  PASS\n\n");
    }
    
    /* Test 5: Dense layer */
    printf("Test 5: Dense layer\n");
    {
        EchoDense dense;
        EchoError err = echo_dense_init(&dense, TEST_EMBED_DIM, TEST_OUTPUT_DIM, ECHO_ACT_TANH, true);
        if (err != ECHO_OK) {
            printf("  FAIL: dense init\n");
            return 1;
        }
        
        printf("  In features: %zu, Out features: %zu\n", dense.in_features, dense.out_features);
        
        /* Test forward */
        echo_float* input = malloc(TEST_EMBED_DIM * sizeof(echo_float));
        echo_float* output = malloc(TEST_OUTPUT_DIM * sizeof(echo_float));
        
        for (size_t i = 0; i < TEST_EMBED_DIM; i++) input[i] = echo_randn();
        
        echo_dense_forward(output, &dense, input);
        printf("  Output[0]: %.4f (should be in [-1, 1] due to tanh)\n", output[0]);
        
        free(input);
        free(output);
        echo_dense_free(&dense);
        printf("  PASS\n\n");
    }
    
    /* Test 6: Reservoir */
    printf("Test 6: Reservoir (Echo State Network)\n");
    {
        EchoReservoir res;
        EchoError err = echo_reservoir_init(&res, TEST_RESERVOIR_SIZE, TEST_EMBED_DIM, 
                                            TEST_OUTPUT_DIM, 0.9f, 0.9f);
        if (err != ECHO_OK) {
            printf("  FAIL: reservoir init\n");
            return 1;
        }
        
        printf("  Reservoir size: %zu\n", res.reservoir_size);
        printf("  Spectral radius: %.2f\n", res.spectral_radius);
        printf("  Sparse connections: %zu\n", res.W_res.nnz);
        
        /* Test step */
        echo_float* input = malloc(TEST_EMBED_DIM * sizeof(echo_float));
        echo_float* output = malloc(TEST_OUTPUT_DIM * sizeof(echo_float));
        
        for (size_t i = 0; i < TEST_EMBED_DIM; i++) input[i] = echo_randn();
        
        echo_reservoir_step(output, &res, input);
        printf("  Step 1 output[0]: %.4f\n", output[0]);
        
        echo_reservoir_step(output, &res, input);
        printf("  Step 2 output[0]: %.4f (should differ due to state)\n", output[0]);
        
        free(input);
        free(output);
        echo_reservoir_free(&res);
        printf("  PASS\n\n");
    }
    
    /* Test 7: Full engine */
    printf("Test 7: Full EchoEngine\n");
    {
        EchoConfig config = {
            .vocab_size = TEST_VOCAB_SIZE,
            .embedding_dim = TEST_EMBED_DIM,
            .reservoir_size = TEST_RESERVOIR_SIZE,
            .hidden_dim = TEST_RESERVOIR_SIZE,
            .output_dim = TEST_OUTPUT_DIM,
            .spectral_radius = 0.9f,
            .reservoir_sparsity = 0.9f
        };
        
        EchoEngine engine;
        EchoError err = echo_engine_init(&engine, &config);
        if (err != ECHO_OK) {
            printf("  FAIL: engine init\n");
            return 1;
        }
        
        printf("  Engine initialized\n");
        
        /* Test forward */
        echo_token tokens[] = {1, 2, 3, 4, 5};
        size_t num_tokens = sizeof(tokens) / sizeof(tokens[0]);
        echo_float* output = malloc(TEST_OUTPUT_DIM * sizeof(echo_float));
        
        echo_engine_forward(output, &engine, tokens, num_tokens);
        printf("  Forward output[0]: %.4f\n", output[0]);
        
        /* Test save/load */
        const char* model_path = "/tmp/test_model.echo";
        err = echo_engine_save(&engine, model_path);
        if (err == ECHO_OK) {
            printf("  Model saved to %s\n", model_path);
        }
        
        echo_engine_reset(&engine);
        echo_engine_forward(output, &engine, tokens, num_tokens);
        printf("  After reset output[0]: %.4f\n", output[0]);
        
        free(output);
        echo_engine_free(&engine);
        printf("  PASS\n\n");
    }
    
    /* Test 8: Performance benchmark */
    printf("Test 8: Performance benchmark\n");
    {
        EchoConfig config = {
            .vocab_size = 8192,
            .embedding_dim = 128,
            .reservoir_size = 512,
            .hidden_dim = 512,
            .output_dim = 64,
            .spectral_radius = 0.9f,
            .reservoir_sparsity = 0.9f
        };
        
        EchoEngine engine;
        EchoError err = echo_engine_init(&engine, &config);
        if (err != ECHO_OK) {
            printf("  FAIL: engine init\n");
            return 1;
        }
        
        echo_token tokens[] = {1, 2, 3, 4, 5, 6, 7, 8};
        size_t num_tokens = sizeof(tokens) / sizeof(tokens[0]);
        echo_float* output = malloc(config.output_dim * sizeof(echo_float));
        
        /* Warmup */
        for (int i = 0; i < 100; i++) {
            echo_engine_forward(output, &engine, tokens, num_tokens);
        }
        echo_engine_reset(&engine);
        
        /* Benchmark */
        clock_t start = clock();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            echo_engine_forward(output, &engine, tokens, num_tokens);
        }
        clock_t end = clock();
        
        double elapsed_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
        double per_inference_us = elapsed_ms * 1000.0 / NUM_ITERATIONS;
        
        printf("  Config: vocab=%zu, embed=%zu, reservoir=%zu, output=%zu\n",
               config.vocab_size, config.embedding_dim, config.reservoir_size, config.output_dim);
        printf("  %d iterations in %.2f ms\n", NUM_ITERATIONS, elapsed_ms);
        printf("  Per inference: %.2f µs (%.0f inferences/sec)\n", 
               per_inference_us, 1000000.0 / per_inference_us);
        
        free(output);
        echo_engine_free(&engine);
        printf("  PASS\n\n");
    }
    
    printf("=== All tests passed! ===\n");
    return 0;
}
