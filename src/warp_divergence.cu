#include <stdio.h>
#include "../include/timer.cuh"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

const int WARP_SIZE = 32;

// all threads take same path
__global__ void no_divergence(float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = idx * 1.0f;
        // All threads do same computation
        for (int i = 0; i < 1000; i++)
        {
            x = x * 1.1f + 0.5f;
        }
        output[idx] = x;
    }
}

// Full divergence - each thread in warp takes different path
__global__ void full_divergence(float *output, int N)
{
    int warp_pos = threadIdx.x % WARP_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x = idx * 1.0f;
        switch (warp_pos)
        {
        case 0:
            for (int i = 0; i < 1000; i++)
                x = x * 1.1f + 0.5f;
            break;
        case 1:
            for (int i = 0; i < 1000; i++)
                x = x * 0.1f + 10.0f;
            break;
        case 2:
            for (int i = 0; i < 1000; i++)
                x = x * 1.3f + 1.5f;
            break;
        case 3:
            for (int i = 0; i < 1000; i++)
                x = x * 1.9f + 5.5f;
            break;
        case 4:
            for (int i = 0; i < 1000; i++)
                x = x * 1.13f + 0.52f;
            break;
        case 5:
            for (int i = 0; i < 1000; i++)
                x = x * 1.1111f + 0.59f;
            break;
        case 6:
            for (int i = 0; i < 1000; i++)
                x = x * 1.9f + 0.555f;
            break;
        case 7:
            for (int i = 0; i < 1000; i++)
                x = x * 9.1f + 9.5f;
            break;
        case 8:
            for (int i = 0; i < 1000; i++)
                x = x * 8.1f + 4.5f;
            break;
        case 9:
            for (int i = 0; i < 1000; i++)
                x = x * 133.1f + 0.225f;
            break;
        case 10:
            for (int i = 0; i < 1000; i++)
                x = x * 1445.1f + 4.5f;
            break;
        case 11:
            for (int i = 0; i < 1000; i++)
                x = x * 132.1f + 1.5f;
            break;
        case 12:
            for (int i = 0; i < 1000; i++)
                x = x * 9.1f + 9.5f;
            break;
        case 13:
            for (int i = 0; i < 1000; i++)
                x = x * 11.1f + 2.5f;
            break;
        case 14:
            for (int i = 0; i < 1000; i++)
                x = x * 13.1f + 4.5f;
            break;
        case 15:
            for (int i = 0; i < 1000; i++)
                x = x * 44.1f + 55.5f;
            break;
        case 16:
            for (int i = 0; i < 1000; i++)
                x = x * 66.1f + 77.5f;
            break;
        case 17:
            for (int i = 0; i < 1000; i++)
                x = x * 99.1f + 88.5f;
            break;
        case 18:
            for (int i = 0; i < 1000; i++)
                x = x * 22.1f + 33.5f;
            break;
        case 19:
            for (int i = 0; i < 1000; i++)
                x = x * 11.1f + 3.5f;
            break;
        case 20:
            for (int i = 0; i < 1000; i++)
                x = x * 67.1f + 98.5f;
            break;
        case 21:
            for (int i = 0; i < 1000; i++)
                x = x * 11.1f + 20.5f;
            break;
        case 22:
            for (int i = 0; i < 1000; i++)
                x = x * 14.1f + 2.5f;
            break;
        case 23:
            for (int i = 0; i < 1000; i++)
                x = x * 4.1f + 1.5f;
            break;
        case 24:
            for (int i = 0; i < 1000; i++)
                x = x * 9.1f + 9.5f;
            break;
        case 25:
            for (int i = 0; i < 1000; i++)
                x = x * 8.1f + 8.5f;
            break;
        case 26:
            for (int i = 0; i < 1000; i++)
                x = x * 7.1f + 7.5f;
            break;
        case 27:
            for (int i = 0; i < 1000; i++)
                x = x * 6.1f + 6.5f;
            break;
        case 28:
            for (int i = 0; i < 1000; i++)
                x = x * 5.1f + 5.5f;
            break;
        case 29:
            for (int i = 0; i < 1000; i++)
                x = x * 4.1f + 4.5f;
            break;
        case 30:
            for (int i = 0; i < 1000; i++)
                x = x * 3.1f + 3.5f;
            break;
        case 31:
            for (int i = 0; i < 1000; i++)
                x = x * 2.1f + 2.5f;
            break;
        }
        output[idx] = x;
    }
}

// Partial divergence
__global__ void partial_divergence(float *output, int N, int divergence_factor)
{
    // TODO: Create controllable divergence
    // divergence_factor = how many different paths (2, 4, 8, 16, 32)

    // [REDACTED - implement partial divergence]
}

void run_warp_divergence()
{
    size_t N = 64 * 1024 * 1024;
    size_t bytes = N * sizeof(float);

    float *d_output;
    cudaMalloc(&d_output, bytes);

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDATimer timer;
    std::vector<float> times;

    // --- 1. no_divergence ---
    for (int i = 0; i < 10; i++)
        no_divergence<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        no_divergence<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto no_divergence_stats = BenchmarkStats::compute(times);
    float baseline_ms = no_divergence_stats.median;
    print_results("no_divergence", baseline_ms * 1000.0f, N);
    times.clear();

    // --- 2. full_divergence ---
    for (int i = 0; i < 10; i++)
        full_divergence<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        full_divergence<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto full_divergence_stats = BenchmarkStats::compute(times);
    float baseline_ms = full_divergence_stats.median;
    print_results("full_divergence", baseline_ms * 1000.0f, N);
    times.clear();

    // std::vector<int> strides = {2, 4, 8, 16, 32, 64};
    // for (int stride : strides)
    // {
    //     // Warmup
    //     strided_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N, stride);
    //     cudaDeviceSynchronize();

    //     for (int i = 0; i < 25; i++)
    //     {
    //         timer.start_timer();
    //         strided_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, d_data, N, stride);
    //         times.push_back(timer.stop_timer());
    //     }
    //     auto strided_stats = BenchmarkStats::compute(times);
    //     print_results("Strided", strided_stats.median * 1000.0f, N, baseline_ms, stride);
    //     times.clear();
    // }

    cudaFree(d_output);
}

int main()
{
    printf("=== Warp Divergence Analysis ===\n");
    run_warp_divergence();
    return 0;
}