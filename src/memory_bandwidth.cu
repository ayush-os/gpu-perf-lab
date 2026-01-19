#include <stdio.h>
#include "../include/timer.cuh"

// Streaming read bandwidth
__global__ void streaming_read(const float4 *input, float4 *output, size_t N_float4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N_float4; i += gridDim.x * blockDim.x)
    {
        float4 val = input[i];

        val.w = val.w * 3.0f + 5.0f;
        val.x = val.x * 3.0f + 5.0f;
        val.y = val.y * 3.0f + 5.0f;
        val.z = val.z * 3.0f + 5.0f;

        output[i] = val;
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
        streaming_read<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N_float4);
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
    benchmark_bandwidth(1024ULL * 1024ULL * 1024ULL, "HBM (1 GB)");

    printf("\nA100 Theoretical Peak HBM Bandwidth: ~1555 GB/s (40GB) or ~2039 GB/s (80GB)\n");

    return 0;
}