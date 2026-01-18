#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/timer.cuh"

// Kernel: Pure register operations
__global__ void register_bandwidth_test(float *out, int iterations, int multiplier)
{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    for (int i = 0; i < iterations; i++)
    {
        a0 = a0 * multiplier + 1.0f;
        a1 = a1 * multiplier + 1.0f;
        a2 = a2 * multiplier + 1.0f;
        a3 = a3 * multiplier + 1.0f;
        a4 = a4 * multiplier + 1.0f;
        a5 = a5 * multiplier + 1.0f;
        a6 = a6 * multiplier + 1.0f;
        a7 = a7 * multiplier + 1.0f;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int main()
{
    // config
    const int WARMUP_ITERS = 10;
    const int TIMING_ITERS = 100;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 432;
    const int KERNEL_ITERS = 10000;
    const int OPS_PER_ITER = 16;

    // Allocate output (just to prevent optimization)
    float *d_output;
    cudaMalloc(&d_output, NUM_BLOCKS * BLOCK_SIZE * sizeof(float));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        register_bandwidth_test<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, KERNEL_ITERS, 2);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < TIMING_ITERS; i++)
    {
        timer.start_timer();
        register_bandwidth_test<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, KERNEL_ITERS, 2);
        float ms = timer.stop_timer();
        times.push_back(ms * 1000); // ms to us
    }

    // Calculate statistics
    auto stats = BenchmarkStats::compute(times);

    printf("=== Register Bandwidth Test ===\n");
    printf("Block size: %d, Iterations: %d\n", BLOCK_SIZE, KERNEL_ITERS);
    stats.print();

    // TODO: Calculate FLOPS
    // OPS_PER_ITER = 2
    // Total operations = BLOCK_SIZE * KERNEL_ITERS * OPS_PER_ITER = 256 * 10000 * 2 = 5,120,000
    // Time (convert us to s) = stats.mean / 1e6
    // 1e9 will convert flops to gflops
    // GFLOPS = (total_ops / time) / 1e9
    auto total_ops = (long long)NUM_BLOCKS * BLOCK_SIZE * KERNEL_ITERS * OPS_PER_ITER;
    auto time = stats.mean / 1e6;
    auto gflops = (total_ops / time) / 1e9;

    printf("GFLOPS: %.2f, TIME: %.6f, TOTAL_OPS: %lld\n", gflops, time, total_ops);

    cudaFree(d_output);
    return 0;
}