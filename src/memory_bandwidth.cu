#include <stdio.h>
#include "../include/timer.cuh"

// Streaming read bandwidth
__global__ void streaming_read(const float4 *__restrict__ input, float4 *__restrict__ output, size_t N_float4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

#pragma unroll 32
    for (int i = idx; i < N_float4; i += stride * 4)
    {
        if (i < N_float4)
        {
            float4 v0 = input[i];
            asm volatile("" : "+f"(v0.x), "+f"(v0.y), "+f"(v0.z), "+f"(v0.w));
            v0.w = v0.w * 3.0f + 5.0f;
            v0.x = v0.x * 3.0f + 5.0f;
            v0.y = v0.y * 3.0f + 5.0f;
            v0.z = v0.z * 3.0f + 5.0f;
            asm volatile("" : : "f"(v0.x), "f"(v0.y), "f"(v0.z), "f"(v0.w) : "memory");
            output[i] = v0;
        }

        if (i + 1 < N_float4)
        {
            float4 v1 = input[i + 1];
            asm volatile("" : "+f"(v1.x), "+f"(v1.y), "+f"(v1.z), "+f"(v1.w));
            v1.w = v1.w * 3.0f + 5.0f;
            v1.x = v1.x * 3.0f + 5.0f;
            v1.y = v1.y * 3.0f + 5.0f;
            v1.z = v1.z * 3.0f + 5.0f;
            asm volatile("" : : "f"(v1.x), "f"(v1.y), "f"(v1.z), "f"(v1.w) : "memory");
            output[i + 1] = v1;
        }

        if (i + 2 < N_float4)
        {
            float4 v2 = input[i + 2];
            asm volatile("" : "+f"(v2.x), "+f"(v2.y), "+f"(v2.z), "+f"(v2.w));
            v2.w = v2.w * 3.0f + 5.0f;
            v2.x = v2.x * 3.0f + 5.0f;
            v2.y = v2.y * 3.0f + 5.0f;
            v2.z = v2.z * 3.0f + 5.0f;
            asm volatile("" : : "f"(v2.x), "f"(v2.y), "f"(v2.z), "f"(v2.w) : "memory");
            output[i + 2] = v2;
        }

        if (i + 3 < N_float4)
        {
            float4 v3 = input[i + 3];
            asm volatile("" : "+f"(v3.x), "+f"(v3.y), "+f"(v3.z), "+f"(v3.w));
            v3.w = v3.w * 3.0f + 5.0f;
            v3.x = v3.x * 3.0f + 5.0f;
            v3.y = v3.y * 3.0f + 5.0f;
            v3.z = v3.z * 3.0f + 5.0f;
            asm volatile("" : : "f"(v3.x), "f"(v3.y), "f"(v3.z), "f"(v3.w) : "memory");
            output[i + 3] = v3;
        }
    }
}

void benchmark_bandwidth(size_t size_bytes, const char *label)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 2160;
    const double MAX_HBM_BANDWIDTH = 2039.0;

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
        streaming_read<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output + (N_float4 / 2), N_float4 / 2);
        float ms = timer.stop_timer();
        times.push_back(ms / 1000.0f); // ms to s
    }

    auto stats = BenchmarkStats::compute(times);

    size_t bytes_transferred = static_cast<size_t>(size_bytes) * 2;
    double bandwidth_gbps = (bytes_transferred / stats.median) / 1e9;
    double utilization = (bandwidth_gbps / MAX_HBM_BANDWIDTH) * 100;

    printf("%-15s: %8.2f GB/s | Util: %6.2f%%\n", label, bandwidth_gbps, utilization);

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