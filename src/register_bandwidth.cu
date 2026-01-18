#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/timer.cuh"

// Kernel: Pure register operations
__global__ void register_bandwidth_test(float *out, int iterations, float mult)
{
    float a0 = 1.0f, a1 = 2.0f, a2 = 3.0f, a3 = 4.0f, a4 = 5.0f, a5 = 6.0f;
    float a6 = 7.0f, a7 = 8.0f, a8 = 9.0f, a9 = 10.0f, a10 = 11.0f, a11 = 12.0f;

    #pragma unroll 32
    for (int i = 0; i < iterations; i++)
    {
        // Inline PTX: fma.rn.f32 dest, src1, src2, src3 (dest = src1 * src2 + src3)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a1) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a2) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a3) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a4) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a5) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a6) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a7) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a8) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a9) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a10) : "f"(mult), "f"(1.0f));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a11) : "f"(mult), "f"(1.0f));
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = (a0 + a1 + a2 + a3 + a4 + a5) + (a6 + a7 + a8 + a9 + a10 + a11);
}

int main()
{
    // config
    const int WARMUP_ITERS = 10;
    const int TIMING_ITERS = 100;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 432;
    const int KERNEL_ITERS = 10000;
    const int OPS_PER_ITER = 24;

    // Allocate output (just to prevent optimization)
    float *d_output;
    cudaMalloc(&d_output, NUM_BLOCKS * BLOCK_SIZE * sizeof(float));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        register_bandwidth_test<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, KERNEL_ITERS, 2.f);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < TIMING_ITERS; i++)
    {
        timer.start_timer();
        register_bandwidth_test<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, KERNEL_ITERS, 2.f);
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