#include <stdio.h>
#include "../include/timer.cuh"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Low register pressure (simple kernel)
__global__ void low_registers(float *data, int N)
{
    // Uses ~10 registers
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = data[idx];
        x = x * 2.0f + 1.0f;
        data[idx] = x;
    }
}

__global__ void high_registers(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float regs[150];

#pragma unroll 1
        for (int i = 0; i < 150; i++)
        {
            regs[i] = data[idx] * (i + 1) * 1.01f;
        }

        float sum = 0.0f;
#pragma unroll 1
        for (int i = 0; i < 150; i++)
        {
            sum += regs[i] * regs[149 - i]; // Cross-dependencies
        }

        data[idx] = sum;
    }
}

// Helper to compute and print stats
void print_results(const char *label, float median_us, float baseline_ms = 0.0f)
{
    float slowdown = (baseline_ms > 0) ? (median_us / (baseline_ms * 1000.0f)) : 1.0f;

    printf("%-22s | Time: %8.2f us | Slowdown: %5.2fx\n",
           label, median_us, slowdown);
}

void run_reg_pressure_analysis()
{
    size_t N = 64 * 1024 * 1024;

    float *d_output;
    cudaMalloc(&d_output, N * sizeof(float));

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDATimer timer;
    std::vector<float> times;

    // --- 1. low_regs ---
    for (int i = 0; i < 10; i++)
        low_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        low_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto low_registers_stats = BenchmarkStats::compute(times);
    float baseline_ms = low_registers_stats.median;
    print_results("low_registers", baseline_ms * 1000.0f, baseline_ms);
    times.clear();

    // --- 2. high_registers ---
    for (int i = 0; i < 10; i++)
        high_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        high_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto high_registers_stats = BenchmarkStats::compute(times);
    print_results("high_registers", high_registers_stats.median * 1000.0f, baseline_ms);
    times.clear();

    cudaFree(d_output);
}

int main()
{
    printf("=== Register Pressure Analysis ===\n");
    run_reg_pressure_analysis();
    return 0;
}