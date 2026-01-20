#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/timer.cuh"

#include <vector>
#include <numeric>   // for std::iota
#include <algorithm> // for std::shuffle
#include <random>    // for std::mt19937 and std::random_device

int *generateCudaIndices(size_t N)
{
    // 1. Generate on Host
    std::vector<int> host_vec(N);
    std::iota(host_vec.begin(), host_vec.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(host_vec.begin(), host_vec.end(), g);

    // 2. Allocate on Device
    int *device_ptr;
    cudaMalloc(&device_ptr, N * sizeof(int));

    // 3. Copy to Device
    cudaMemcpy(device_ptr, host_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    return device_ptr;
}

// Coalesced access (best case)
__global__ void coalesced_access(float *output, float *data, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Adjacent threads access adjacent memory
        // Thread 0 → data[0]
        // Thread 1 → data[1]
        // Thread 2 → data[2]
        // This is GOOD
        output[idx] = data[idx] * 2.0f;
    }
}

// Strided access (worse performance)
__global__ void strided_access(float *output, float *data, size_t N, int stride)
{
    // Instead of adjacent threads accessing adjacent memory,
    // they access memory 'stride' apart
    //
    // Thread 0 → data[0]
    // Thread 1 → data[stride]
    // Thread 2 → data[2*stride]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int access_idx = idx * stride;
    if (access_idx < N)
    {
        output[access_idx] = data[access_idx] * 2.0f;
    }
}

// Random access (worst case)
__global__ void random_access(float *output, float *data, int *indices, size_t N)
{
    // Thread i accesses data[indices[i]]
    // Where indices is a pre-shuffled array
    //
    // This defeats all caching and coalescing

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        output[idx] = data[indices[idx]] * 2.0f;
    }
}

void compare_access_patterns()
{
    size_t N = 64 * 1024 * 1024; // 256 MB
    size_t bytes = N * sizeof(float);

    float *d_data;
    float *d_output;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemset(d_data, 0, bytes);

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Benchmark each access pattern
    // 1. Coalesced
    // 2. Strided (try stride = 2, 16, 32, 64)
    // 3. Random
    //
    // For each, measure:
    // - Time
    // - Achieved bandwidth
    // - Slowdown vs coalesced

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        coalesced_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        coalesced_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N);
        float ms = timer.stop_timer();
        times.push_back(ms * 1000); // ms to us
    }

    // Calculate statistics
    auto stats = BenchmarkStats::compute(times);

    printf("coalesced | time: %f | achieved bandwidth: %f", stats.median, 0.0f);
    times.clear();

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        strided_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N, 16);
    }
    cudaDeviceSynchronize();

    std::vector<int> strides = {2, 16, 32, 64};

    for (auto &stride : strides)
    {
        for (int i = 0; i < 25; i++)
        {
            timer.start_timer();
            strided_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N, stride);
            float ms = timer.stop_timer();
            times.push_back(ms * 1000); // ms to us
        }

        // Calculate statistics
        stats = BenchmarkStats::compute(times);

        printf("coalesced | time: %f | achieved bandwidth: %f | stride: %d", stats.median, 0.0f, stride);
        times.clear();
    }

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
        float ms = timer.stop_timer();
        times.push_back(ms * 1000); // ms to us
    }

    stats = BenchmarkStats::compute(times);
    printf("randomized | time: %f | achieved bandwidth: %f", stats.median, 0.0f);

    cudaFree(d_data);
    cudaFree(d_output);
}

int main()
{
    printf("=== Memory Access Pattern Analysis ===\n\n");
    compare_access_patterns();
    return 0;
}