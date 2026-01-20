#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/timer.cuh"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Helper to compute and print stats
void print_results(const char *label, float median_us, size_t N, float baseline_ms = 0.0f, int stride = 0)
{
    // Bandwidth = (Read Bytes + Write Bytes) / Time
    // 2 * N * 4 bytes / (median_us / 1e6) / 1e9 = GB/s
    double total_bytes = 2.0 * N * sizeof(float);
    double seconds = median_us / 1e6;
    double gb_s = (total_bytes / seconds) / 1e9;

    float slowdown = (baseline_ms > 0) ? (median_us / (baseline_ms * 1000.0f)) : 1.0f;

    if (stride > 0)
    {
        printf("%-12s (stride %2d) | Time: %8.2f us | BW: %7.2f GB/s | Slowdown: %5.2fx\n",
               label, stride, median_us, gb_s, slowdown);
    }
    else
    {
        printf("%-22s | Time: %8.2f us | BW: %7.2f GB/s | Slowdown: %5.2fx\n",
               label, median_us, gb_s, slowdown);
    }
}

int *generateCudaIndices(size_t N)
{
    std::vector<int> host_vec(N);
    std::iota(host_vec.begin(), host_vec.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(host_vec.begin(), host_vec.end(), g);

    int *device_ptr;
    cudaMalloc(&device_ptr, N * sizeof(int));
    cudaMemcpy(device_ptr, host_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    return device_ptr;
}

__global__ void coalesced_access(float *output, float *data, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = data[idx] * 2.0f;
    }
}

// Strided access
__global__ void strided_access(float *output, float *data, size_t N, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        size_t strided_idx = (size_t(idx) * stride) % N;
        output[strided_idx] = data[strided_idx] * 2.0f;
    }
}

__global__ void random_access(float *output, float *data, int *indices, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int target = indices[idx];
        output[target] = data[target] * 2.0f;
    }
}

void compare_access_patterns()
{
    size_t N = 64 * 1024 * 1024;
    size_t bytes = N * sizeof(float);

    float *d_data, *d_output;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemset(d_data, 0, bytes);

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDATimer timer;
    std::vector<float> times;

    // --- 1. COALESCED ---
    for (int i = 0; i < 10; i++)
        coalesced_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        coalesced_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N);
        times.push_back(timer.stop_timer());
    }
    auto coalesced_stats = BenchmarkStats::compute(times);
    float baseline_ms = coalesced_stats.median;
    print_results("Coalesced", baseline_ms * 1000.0f, N);
    times.clear();

    // --- 2. STRIDED ---
    std::vector<int> strides = {2, 4, 8, 16, 32, 64};
    for (int stride : strides)
    {
        // Warmup
        strided_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N, stride);
        cudaDeviceSynchronize();

        for (int i = 0; i < 25; i++)
        {
            timer.start_timer();
            strided_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N, stride);
            times.push_back(timer.stop_timer());
        }
        auto strided_stats = BenchmarkStats::compute(times);
        print_results("Strided", strided_stats.median * 1000.0f, N, baseline_ms, stride);
        times.clear();
    }

    // --- 3. RANDOM ---
    int *indices = generateCudaIndices(N);

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        random_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, indices, N);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        random_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, indices, N);
        times.push_back(timer.stop_timer());
    }
    auto random_stats = BenchmarkStats::compute(times);
    print_results("Randomized", random_stats.median * 1000.0f, N, baseline_ms);

    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(indices);
}

int main()
{
    printf("=== Memory Access Pattern Analysis ===\n");
    printf("N: 64M elements (256MB per array)\n\n");
    compare_access_patterns();
    return 0;
}