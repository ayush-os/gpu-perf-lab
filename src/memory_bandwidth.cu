#include <stdio.h>
#include "../include/timer.cuh"

// Streaming read bandwidth
__global__ void streaming_read(const float4 *__restrict__ input,
                               float4 *__restrict__ output,
                               size_t N_float4)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

#pragma unroll 32
    for (size_t i = idx; i < N_float4; i += stride)
    {
        float4 v = input[i];

        // Use asm to force the load and math to actually happen
        asm volatile("" : "+f"(v.x), "+f"(v.y), "+f"(v.z), "+f"(v.w));

        v.x = v.x * 3.0f + 5.0f;
        v.y = v.y * 3.0f + 5.0f;
        v.z = v.z * 3.0f + 5.0f;
        v.w = v.w * 3.0f + 5.0f;

        asm volatile("" : : "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");

        output[i] = v;
    }
}

void benchmark_bandwidth(size_t size_bytes)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 16384;

    size_t N_float4 = size_bytes / sizeof(float4);

    float4 *d_input, *d_output;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);

    cudaMemset(d_input, 1, size_bytes);

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        streaming_read<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output + (N_float4 / 2), N_float4 / 2);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < 100; i++)
    {
        timer.start_timer();
        streaming_read<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N_float4);
        float ms = timer.stop_timer();
        times.push_back(ms / 1000.0f); // ms to s
    }

    auto stats = BenchmarkStats::compute(times);

    double bytes_transferred = (double)size_bytes * 2.0;
    double bandwidth_gbps = (bytes_transferred / stats.median) / 1e9;

    printf("%-15s: %8.2f GB/s | Util: %6.2f%%\n", label, bandwidth_gbps, (bandwidth_gbps / 2039.0) * 100.0);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    printf("=== Memory Bandwidth Measurement ===\n\n");

    benchmark_bandwidth(64 * 1024, "L1 Cache (64 KB)");
    benchmark_bandwidth(4 * 1024 * 1024, "L2 Cache (4 MB)");
    benchmark_bandwidth(8 * 1024ULL * 1024ULL * 1024ULL, "HBM (8 GB)");

    printf("\nA100 Theoretical Peak HBM Bandwidth: ~1555 GB/s (40GB) or ~2039 GB/s (80GB)\n");

    return 0;
}