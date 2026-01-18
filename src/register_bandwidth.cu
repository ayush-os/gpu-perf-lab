#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/timer.cuh"

// Kernel: Pure register operations
__global__ void register_bandwidth_test(float *output, int iterations)
{
    float acc = 0.1f;
    for (int i = 0; i < iterations; i++)
    {
        acc += 1.01 * acc;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = acc;
}

int main()
{
    // config
    const int WARMUP_ITERS = 10;
    const int TIMING_ITERS = 100;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 432;
    const int KERNEL_ITERS = 10000;
    const int OPS_PER_ITER = 2;

    // Allocate output (just to prevent optimization)
    float *d_output;
    cudaMalloc(&d_output, NUM_BLOCKS * BLOCK_SIZE * sizeof(float));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        register_bandwidth_test<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, KERNEL_ITERS);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < TIMING_ITERS; i++)
    {
        timer.start_timer();
        register_bandwidth_test<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, KERNEL_ITERS);
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

    printf("GFLOPS: %.2f, TIME: %.6f, TOTAL_OPS: %d\n", gflops, time, total_ops);

    cudaFree(d_output);
    return 0;
}