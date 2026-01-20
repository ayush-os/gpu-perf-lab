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
        // 1. Initialize all variables to ensure they occupy registers immediately
        float r000 = data[idx];
        float r001 = r000 * 1.01f + 0.1f;
        float r002 = r001 * 1.01f + 0.2f;
        float r003 = r002 * 1.01f + 0.3f;
        float r004 = r003 * 1.01f + 0.4f;
        float r005 = r004 * 1.01f + 0.5f;
        float r006 = r005 * 1.01f + 0.6f;
        float r007 = r006 * 1.01f + 0.7f;
        float r008 = r007 * 1.01f + 0.8f;
        float r009 = r008 * 1.01f + 0.9f;
        float r010 = r009 * 1.01f + 1.0f;
        float r011 = r010 * 1.01f + 1.1f;
        float r012 = r011 * 1.01f + 1.2f;
        float r013 = r012 * 1.01f + 1.3f;
        float r014 = r013 * 1.01f + 1.4f;
        float r015 = r014 * 1.01f + 1.5f;
        float r016 = r015 * 1.01f + 1.6f;
        float r017 = r016 * 1.01f + 1.7f;
        float r018 = r017 * 1.01f + 1.8f;
        float r019 = r018 * 1.01f + 1.9f;
        float r020 = r019 * 1.01f + 2.0f;
        float r021 = r020 * 1.01f + 2.1f;
        float r022 = r021 * 1.01f + 2.2f;
        float r023 = r022 * 1.01f + 2.3f;
        float r024 = r023 * 1.01f + 2.4f;
        float r025 = r024 * 1.01f + 2.5f;
        float r026 = r025 * 1.01f + 2.6f;
        float r027 = r026 * 1.01f + 2.7f;
        float r028 = r027 * 1.01f + 2.8f;
        float r029 = r028 * 1.01f + 2.9f;
        float r030 = r029 * 1.01f + 3.0f;
        float r031 = r030 * 1.01f + 3.1f;
        float r032 = r031 * 1.01f + 3.2f;
        float r033 = r032 * 1.01f + 3.3f;
        float r034 = r033 * 1.01f + 3.4f;
        float r035 = r034 * 1.01f + 3.5f;
        float r036 = r035 * 1.01f + 3.6f;
        float r037 = r036 * 1.01f + 3.7f;
        float r038 = r037 * 1.01f + 3.8f;
        float r039 = r038 * 1.01f + 3.9f;
        float r040 = r039 * 1.01f + 4.0f;
        float r041 = r040 * 1.01f + 4.1f;
        float r042 = r041 * 1.01f + 4.2f;
        float r043 = r042 * 1.01f + 4.3f;
        float r044 = r043 * 1.01f + 4.4f;
        float r045 = r044 * 1.01f + 4.5f;
        float r046 = r045 * 1.01f + 4.6f;
        float r047 = r046 * 1.01f + 4.7f;
        float r048 = r047 * 1.01f + 4.8f;
        float r049 = r048 * 1.01f + 4.9f;
        float r050 = r049 * 1.01f + 5.0f;
        float r051 = r050 * 1.01f + 5.1f;
        float r052 = r051 * 1.01f + 5.2f;
        float r053 = r052 * 1.01f + 5.3f;
        float r054 = r053 * 1.01f + 5.4f;
        float r055 = r054 * 1.01f + 5.5f;
        float r056 = r055 * 1.01f + 5.6f;
        float r057 = r056 * 1.01f + 5.7f;
        float r058 = r057 * 1.01f + 5.8f;
        float r059 = r058 * 1.01f + 5.9f;
        float r060 = r059 * 1.01f + 6.0f;
        float r061 = r060 * 1.01f + 6.1f;
        float r062 = r061 * 1.01f + 6.2f;
        float r063 = r062 * 1.01f + 6.3f;
        float r064 = r063 * 1.01f + 6.4f;
        float r065 = r064 * 1.01f + 6.5f;
        float r066 = r065 * 1.01f + 6.6f;
        float r067 = r066 * 1.01f + 6.7f;
        float r068 = r067 * 1.01f + 6.8f;
        float r069 = r068 * 1.01f + 6.9f;
        float r070 = r069 * 1.01f + 7.0f;
        float r071 = r070 * 1.01f + 7.1f;
        float r072 = r071 * 1.01f + 7.2f;
        float r073 = r072 * 1.01f + 7.3f;
        float r074 = r073 * 1.01f + 7.4f;
        float r075 = r074 * 1.01f + 7.5f;
        float r076 = r075 * 1.01f + 7.6f;
        float r077 = r076 * 1.01f + 7.7f;
        float r078 = r077 * 1.01f + 7.8f;
        float r079 = r078 * 1.01f + 7.9f;
        float r080 = r079 * 1.01f + 8.0f;
        float r081 = r080 * 1.01f + 8.1f;
        float r082 = r081 * 1.01f + 8.2f;
        float r083 = r082 * 1.01f + 8.3f;
        float r084 = r083 * 1.01f + 8.4f;
        float r085 = r084 * 1.01f + 8.5f;
        float r086 = r085 * 1.01f + 8.6f;
        float r087 = r086 * 1.01f + 8.7f;
        float r088 = r087 * 1.01f + 8.8f;
        float r089 = r088 * 1.01f + 8.9f;
        float r090 = r089 * 1.01f + 9.0f;
        float r091 = r090 * 1.01f + 9.1f;
        float r092 = r091 * 1.01f + 9.2f;
        float r093 = r092 * 1.01f + 9.3f;
        float r094 = r093 * 1.01f + 9.4f;
        float r095 = r094 * 1.01f + 9.5f;
        float r096 = r095 * 1.01f + 9.6f;
        float r097 = r096 * 1.01f + 9.7f;
        float r098 = r097 * 1.01f + 9.8f;
        float r099 = r098 * 1.01f + 9.9f;
        float r100 = r099 * 1.01f + 10.0f;
        float r101 = r100 * 1.01f + 10.1f;
        float r102 = r101 * 1.01f + 10.2f;
        float r103 = r102 * 1.01f + 10.3f;
        float r104 = r103 * 1.01f + 10.4f;
        float r105 = r104 * 1.01f + 10.5f;
        float r106 = r105 * 1.01f + 10.6f;
        float r107 = r106 * 1.01f + 10.7f;
        float r108 = r107 * 1.01f + 10.8f;
        float r109 = r108 * 1.01f + 10.9f;
        float r110 = r109 * 1.01f + 11.0f;
        float r111 = r110 * 1.01f + 11.1f;
        float r112 = r111 * 1.01f + 11.2f;
        float r113 = r112 * 1.01f + 11.3f;
        float r114 = r113 * 1.01f + 11.4f;
        float r115 = r114 * 1.01f + 11.5f;
        float r116 = r115 * 1.01f + 11.6f;
        float r117 = r116 * 1.01f + 11.7f;
        float r118 = r117 * 1.01f + 11.8f;
        float r119 = r118 * 1.01f + 11.9f;
        float r120 = r119 * 1.01f + 12.0f;
        float r121 = r120 * 1.01f + 12.1f;
        float r122 = r121 * 1.01f + 12.2f;
        float r123 = r122 * 1.01f + 12.3f;
        float r124 = r123 * 1.01f + 12.4f;
        float r125 = r124 * 1.01f + 12.5f;
        float r126 = r125 * 1.01f + 12.6f;
        float r127 = r126 * 1.01f + 12.7f;
        float r128 = r127 * 1.01f + 12.8f;
        float r129 = r128 * 1.01f + 12.9f;
        float r130 = r129 * 1.01f + 13.0f;
        float r131 = r130 * 1.01f + 13.1f;
        float r132 = r131 * 1.01f + 13.2f;
        float r133 = r132 * 1.01f + 13.3f;
        float r134 = r133 * 1.01f + 13.4f;
        float r135 = r134 * 1.01f + 13.5f;
        float r136 = r135 * 1.01f + 13.6f;
        float r137 = r136 * 1.01f + 13.7f;
        float r138 = r137 * 1.01f + 13.8f;
        float r139 = r138 * 1.01f + 13.9f;
        float r140 = r139 * 1.01f + 14.0f;
        float r141 = r140 * 1.01f + 14.1f;
        float r142 = r141 * 1.01f + 14.2f;
        float r143 = r142 * 1.01f + 14.3f;
        float r144 = r143 * 1.01f + 14.4f;
        float r145 = r144 * 1.01f + 14.5f;
        float r146 = r145 * 1.01f + 14.6f;
        float r147 = r146 * 1.01f + 14.7f;
        float r148 = r147 * 1.01f + 14.8f;
        float r149 = r148 * 1.01f + 14.9f;

        // 2. Perform the reduction to force the compiler to keep all variables "live"
        data[idx] = r000 + r001 + r002 + r003 + r004 + r005 + r006 + r007 + r008 + r009 +
                    r010 + r011 + r012 + r013 + r014 + r015 + r016 + r017 + r018 + r019 +
                    r020 + r021 + r022 + r023 + r024 + r025 + r026 + r027 + r028 + r029 +
                    r030 + r031 + r032 + r033 + r034 + r035 + r036 + r037 + r038 + r039 +
                    r040 + r041 + r042 + r043 + r044 + r045 + r046 + r047 + r048 + r049 +
                    r050 + r051 + r052 + r053 + r054 + r055 + r056 + r057 + r058 + r059 +
                    r060 + r061 + r062 + r063 + r064 + r065 + r066 + r067 + r068 + r069 +
                    r070 + r071 + r072 + r073 + r074 + r075 + r076 + r077 + r078 + r079 +
                    r080 + r081 + r082 + r083 + r084 + r085 + r086 + r087 + r088 + r089 +
                    r090 + r091 + r092 + r093 + r094 + r095 + r096 + r097 + r098 + r099 +
                    r100 + r101 + r102 + r103 + r104 + r105 + r106 + r107 + r108 + r109 +
                    r110 + r111 + r112 + r113 + r114 + r115 + r116 + r117 + r118 + r119 +
                    r120 + r121 + r122 + r123 + r124 + r125 + r126 + r127 + r128 + r129 +
                    r130 + r131 + r132 + r133 + r134 + r135 + r136 + r137 + r138 + r139 +
                    r140 + r141 + r142 + r143 + r144 + r145 + r146 + r147 + r148 + r149;
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