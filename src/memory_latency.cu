#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include "../include/timer.cuh"

// Prepare pointer chain on host, copy to device
void prepare_pointer_chase(int **h_chain, int **d_chain, size_t size_bytes, int stride)
{
    size_t num_elems = size_bytes / sizeof(int);
    *h_chain = (int *)malloc(size_bytes);

    // 1. Create a sequence of indices [0, 1, 2, ..., n-1]
    std::vector<int> indices(num_elems);
    std::iota(indices.begin(), indices.end(), 0);

    // 2. Shuffle the indices to create a random visitation order
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // 3. Link the indices in the shuffled order to form one giant circle
    // If shuffled indices are [3, 0, 2, 1], then:
    // h_chain[3] = 0, h_chain[0] = 2, h_chain[2] = 1, h_chain[1] = 3
    for (size_t i = 0; i < num_elems - 1; i++)
    {
        (*h_chain)[indices[i]] = indices[i + 1];
    }
    // Close the loop
    (*h_chain)[indices[num_elems - 1]] = indices[0];

    cudaMalloc((void **)d_chain, size_bytes);
    cudaMemcpy(*d_chain, *h_chain, size_bytes, cudaMemcpyHostToDevice);
}

__global__ void pointer_chase_kernel(int *chain, int *output, int chase_count)
{
    int cur = chain[threadIdx.x];

    for (int i = 0; i < chase_count; i++)
    {
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
        cur = chain[cur];
    }

    output[threadIdx.x] = cur;
}

void benchmark_latency(size_t size_bytes, int stride, const char *label)
{
    const int WARMUP_ITERS = 10;
    const int TIMING_ITERS = 100;
    const int BLOCK_SIZE = 1;
    const int NUM_BLOCKS = 1;
    const int CHASE_COUNT = 10000;

    int *d_output;
    cudaMalloc(&d_output, NUM_BLOCKS * BLOCK_SIZE * sizeof(int));

    int *h_chain;
    int *d_chain;
    prepare_pointer_chase(&h_chain, &d_chain, size_bytes, stride);

    // warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        pointer_chase_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_chain, d_output, CHASE_COUNT);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < TIMING_ITERS; i++)
    {
        timer.start_timer();
        pointer_chase_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_chain, d_output, CHASE_COUNT);
        float ms = timer.stop_timer();
        times.push_back(ms * 1000000); // ms to ns
    }

    auto stats = BenchmarkStats::compute(times);

    printf("%s (size=%zu KB, stride=%d): %.2f ns per access\n",
           label, size_bytes / 1024, stride, stats.median / CHASE_COUNT);
    stats.print();

    free(h_chain);
    cudaFree(d_chain);
    cudaFree(d_output);
    return;
}

int main()
{
    printf("=== Memory Latency Measurement ===\n\n");

    // L1 cache: ~192 KB on A100
    // 64 KB to hit L1
    benchmark_latency(64 * 1024, 1, "L1 Cache");
    benchmark_latency(64 * 1024, 16, "L1 Cache");
    benchmark_latency(64 * 1024, 64, "L1 Cache");
    benchmark_latency(64 * 1024, 256, "L1 Cache");

    // L2 cache: 40 MB on A100
    // 4 MB to hit L2
    benchmark_latency(4 * 1024 * 1024, 1, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 16, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 64, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 256, "L2 Cache");

    // HBM: 40/80 GB on A100
    // 1GB to hit HBM
    benchmark_latency(1000 * 1024 * 1024, 1, "HBM");
    benchmark_latency(1000 * 1024 * 1024, 16, "HBM");
    benchmark_latency(1000 * 1024 * 1024, 64, "HBM");
    benchmark_latency(1000 * 1024 * 1024, 256, "HBM");

    return 0;
}
