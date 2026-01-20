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
        float x = data[idx];
        float r0 = x * 1.1f + 0.5f;
        float r1 = x * 2.1f + 1.5f;
        float r2 = x * 3.1f + 2.5f;
        float r3 = x * 4.1f + 3.5f;
        float r4 = x * 5.1f + 4.5f;
        float r5 = x * 6.1f + 5.5f;
        float r6 = x * 7.1f + 6.5f;
        float r7 = x * 8.1f + 7.5f;
        float r8 = x * 9.1f + 8.5f;
        float r9 = x * 10.1f + 9.5f;
        float r10 = x * 11.1f + 10.5f;
        float r11 = x * 12.1f + 11.5f;
        float r12 = x * 13.1f + 12.5f;
        float r13 = x * 14.1f + 13.5f;
        float r14 = x * 15.1f + 14.5f;
        float r15 = x * 16.1f + 15.5f;
        float r16 = x * 17.1f + 16.5f;
        float r17 = x * 18.1f + 17.5f;
        float r18 = x * 19.1f + 18.5f;
        float r19 = x * 20.1f + 19.5f;
        float r20 = x * 21.1f + 20.5f;
        float r21 = x * 22.1f + 21.5f;
        float r22 = x * 23.1f + 22.5f;
        float r23 = x * 24.1f + 23.5f;
        float r24 = x * 25.1f + 24.5f;
        float r25 = x * 26.1f + 25.5f;
        float r26 = x * 27.1f + 26.5f;
        float r27 = x * 28.1f + 27.5f;
        float r28 = x * 29.1f + 28.5f;
        float r29 = x * 30.1f + 29.5f;
        float r30 = x * 31.1f + 30.5f;
        float r31 = x * 32.1f + 31.5f;
        float r32 = x * 33.1f + 32.5f;
        float r33 = x * 34.1f + 33.5f;
        float r34 = x * 35.1f + 34.5f;
        float r35 = x * 36.1f + 35.5f;
        float r36 = x * 37.1f + 36.5f;
        float r37 = x * 38.1f + 37.5f;
        float r38 = x * 39.1f + 38.5f;
        float r39 = x * 40.1f + 39.5f;
        float r40 = x * 41.1f + 40.5f;
        float r41 = x * 42.1f + 41.5f;
        float r42 = x * 43.1f + 42.5f;
        float r43 = x * 44.1f + 43.5f;
        float r44 = x * 45.1f + 44.5f;
        float r45 = x * 46.1f + 45.5f;
        float r46 = x * 47.1f + 46.5f;
        float r47 = x * 48.1f + 47.5f;
        float r48 = x * 49.1f + 48.5f;
        float r49 = x * 50.1f + 49.5f;
        float r50 = x * 51.1f + 50.5f;
        float r51 = x * 52.1f + 51.5f;
        float r52 = x * 53.1f + 52.5f;
        float r53 = x * 54.1f + 53.5f;
        float r54 = x * 55.1f + 54.5f;
        float r55 = x * 56.1f + 55.5f;
        float r56 = x * 57.1f + 56.5f;
        float r57 = x * 58.1f + 57.5f;
        float r58 = x * 59.1f + 58.5f;
        float r59 = x * 60.1f + 59.5f;
        float r60 = x * 61.1f + 60.5f;
        float r61 = x * 62.1f + 61.5f;
        float r62 = x * 63.1f + 62.5f;
        float r63 = x * 64.1f + 63.5f;
        float r64 = x * 65.1f + 64.5f;
        float r65 = x * 66.1f + 65.5f;
        float r66 = x * 67.1f + 66.5f;
        float r67 = x * 68.1f + 67.5f;
        float r68 = x * 69.1f + 68.5f;
        float r69 = x * 70.1f + 69.5f;
        float r70 = x * 71.1f + 70.5f;
        float r71 = x * 72.1f + 71.5f;
        float r72 = x * 73.1f + 72.5f;
        float r73 = x * 74.1f + 73.5f;
        float r74 = x * 75.1f + 74.5f;
        float r75 = x * 76.1f + 75.5f;
        float r76 = x * 77.1f + 76.5f;
        float r77 = x * 78.1f + 77.5f;
        float r78 = x * 79.1f + 78.5f;
        float r79 = x * 80.1f + 79.5f;

        float result = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25 + r26 + r27 + r28 + r29 + r30 + r31 + r32 + r33 + r34 + r35 + r36 + r37 + r38 + r39 + r40 + r41 + r42 + r43 + r44 + r45 + r46 + r47 + r48 + r49 + r50 + r51 + r52 + r53 + r54 + r55 + r56 + r57 + r58 + r59 + r60 + r61 + r62 + r63 + r64 + r65 + r66 + r67 + r68 + r69 + r70 + r71 + r72 + r73 + r74 + r75 + r76 + r77 + r78 + r79;

        data[idx] = result;
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