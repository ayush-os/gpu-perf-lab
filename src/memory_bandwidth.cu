#include <stdio.h>
#include "../include/timer.cuh"

// Streaming read bandwidth
__global__ void streaming_read(const float *input, float *output, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
    {
        output[i] = input[i] * 3 + 5;
    }
}

// Streaming write bandwidth
__global__ void streaming_write(float *output, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
    {
        output[i] = i * 3 + 5;
    }
}

void benchmark_bandwidth(size_t size_bytes, const char *label)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 216;
    const int MAX_HBM_BANDWIDTH = 2039;

    size_t N = size_bytes / sizeof(float);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);

    cudaMemset(d_input, 1, size_bytes);

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        streaming_read<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < 100; i++)
    {
        timer.start_timer();
        streaming_read<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
        float ms = timer.stop_timer();
        times.push_back(ms / 1000); // ms to s
    }

    auto stats = BenchmarkStats::compute(times);

    int bytes_transferred = size_bytes * 2;
    int bandwidth_gbps = (bytes_transferred / stats.median) / 1e9;
    double utilization = (static_cast<double>(bandwidth_gbps) / MAX_HBM_BANDWIDTH) * 100;

    printf("%s: %.2f GB/s; util: %.2f\n", label, bandwidth_gbps, utilization);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    printf("=== Memory Bandwidth Measurement ===\n\n");

    benchmark_bandwidth(64 * 1024, "L1 Cache (64 KB)");
    benchmark_bandwidth(4 * 1024 * 1024, "L2 Cache (4 MB)");
    benchmark_bandwidth(1024 * 1024 * 1024, "HBM (1 GB)");

    printf("\nA100 Theoretical Peak HBM Bandwidth: ~1555 GB/s (40GB) or ~2039 GB/s (80GB)\n");
    // TODO
    printf("Your achieved percentage: [calculate this]\n");

    return 0;
}