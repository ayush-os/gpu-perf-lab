#include <stdio.h>
#include "../include/timer.cuh"

// No bank conflicts (sequential access)
__global__ void no_conflicts(float *output, int N)
{
    __shared__ float smem[256];

    // Each thread accesses different bank
    // Bank = (address / 4) % 32
    // Thread i accesses smem[i], so banks are 0,1,2,...,31,0,1,2,...
    smem[threadIdx.x] = threadIdx.x * 1.0f;
    __syncthreads();

    // Read back
    float val = smem[threadIdx.x];
    output[blockIdx.x * blockDim.x + threadIdx.x] = val + 0.000001f * (float)threadIdx.x;
}

// Bank conflicts (stride access)
__global__ void with_conflicts(float *output, int stride, int N)
{
    // Access pattern: smem[threadIdx.x * stride]
    //
    // - stride = 1: no conflicts (different banks)
    // - stride = 2: 2-way conflicts
    // - stride = 32: 32-way conflicts (worst case!)
    // - stride = 33: no conflicts (coprime with 32)

    __shared__ float smem[256];

    smem[threadIdx.x] = threadIdx.x * 1.0f;
    __syncthreads();

    float val = ((volatile float*)smem)[threadIdx.x * stride];
    output[blockIdx.x * blockDim.x + threadIdx.x] = val + 0.000001f * (float)threadIdx.x;
}

// Broadcast (special case - no conflict)
__global__ void broadcast_access(float *output, int N)
{
    __shared__ float smem[256];

    smem[threadIdx.x] = threadIdx.x * 1.0f;
    __syncthreads();

    float val = smem[0];
    output[blockIdx.x * blockDim.x + threadIdx.x] = val + 0.000001f * (float)threadIdx.x;
}

void print_results(const char *label, float median_us, size_t N, float baseline_ms = 0.0f, int stride = 0)
{
    // Total bytes moved across the bus (Global Memory Writes)
    // Note: Shared memory traffic isn't counted in global BW,
    // but this helps show the performance impact on the overall kernel.
    double total_bytes = (double)N * sizeof(float);
    double seconds = median_us / 1e6;
    double gb_s = (total_bytes / seconds) / 1e9;

    // Calculate how much slower this is than the "No Conflict" case
    float slowdown = (baseline_ms > 0) ? (median_us / (baseline_ms * 1000.0f)) : 1.0f;

    if (stride > 0)
    {
        // Highlight specific characteristics of common strides
        const char *note = "";
        if (stride == 32)
            note = " (32-way conflict!)";
        if (stride == 33)
            note = " (Conflict-free)";

        printf("%-15s | Time: %8.2f us | BW: %7.2f GB/s | Slowdown: %5.2fx%s\n",
               label, median_us, gb_s, slowdown, note);
    }
    else
    {
        printf("%-15s | Time: %8.2f us | BW: %7.2f GB/s | Slowdown: %5.2fx\n",
               label, median_us, gb_s, slowdown);
    }
}

void benchmark_shared_memory()
{
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = N / BLOCK_SIZE;

    float *d_output;
    cudaMalloc(&d_output, N * sizeof(float));

    CUDATimer timer;
    std::vector<float> times;

    // 1. Baseline: No Conflicts
    // Warmup
    no_conflicts<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 100; i++)
    {
        timer.start_timer();
        no_conflicts<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto baseline_stats = BenchmarkStats::compute(times);
    float baseline_ms = baseline_stats.median;
    print_results("No Conflicts", baseline_ms * 1000.0f, N);
    times.clear();

    // 2. Broadcast (Special Case)
    broadcast_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 100; i++)
    {
        timer.start_timer();
        broadcast_access<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto broadcast_stats = BenchmarkStats::compute(times);
    print_results("Broadcast", broadcast_stats.median * 1000.0f, N, baseline_ms);
    times.clear();

    // 3. Strided Access (The Conflict Tests)
    std::vector<int> strides = {1, 2, 4, 8, 16, 32, 33, 64};

    printf("\n--- Testing Strided Access (Bank Conflicts) ---\n");
    for (int stride : strides)
    {
        // Warmup
        with_conflicts<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, stride, N);
        cudaDeviceSynchronize();

        for (int i = 0; i < 100; i++)
        {
            timer.start_timer();
            with_conflicts<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, stride, N);
            times.push_back(timer.stop_timer());
        }

        auto strided_stats = BenchmarkStats::compute(times);
        char label[20];
        sprintf(label, "Stride %d", stride);

        // Note: Stride 32 is 32-way conflict (slowest), Stride 33 is 0-way (fastest)
        print_results(label, strided_stats.median * 1000.0f, N, baseline_ms, stride);
        times.clear();
    }

    cudaFree(d_output);
}

int main()
{
    printf("=== Shared Memory Bank Conflicts ===\n\n");
    benchmark_shared_memory();
    return 0;
}