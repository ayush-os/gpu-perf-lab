#include <stdio.h>
#include "../include/timer.cuh"

// No bank conflicts (sequential access)
__global__ void no_conflicts(float *output, int N)
{
    __shared__ float smem[256];

    float acc = 0.0f;
    smem[threadIdx.x] = threadIdx.x * 1.0f;
    __syncthreads();

#pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        acc += smem[threadIdx.x];
        __syncthreads();
    }

    output[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Bank conflicts (stride access)
__global__ void with_conflicts(float *output, int stride, int N)
{
    __shared__ float smem[8192]; // Larger to handle stride * 256

    // Initialize all of shared memory we might touch
    for (int i = threadIdx.x; i < 8192; i += blockDim.x)
        smem[i] = i * 1.0f;
    __syncthreads();

    float acc = 0.0f;

#pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        int idx = (threadIdx.x * stride) % 8192;
        acc += smem[idx];
        __syncthreads();
    }

    output[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Broadcast (special case - no conflict)
__global__ void broadcast_access(float *output, int N)
{
    __shared__ float smem[256];

    smem[threadIdx.x] = threadIdx.x * 1.0f;
    __syncthreads();

    float acc = 0.0f;

#pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        acc += smem[0];
        __syncthreads();
    }

    output[blockIdx.x * blockDim.x + threadIdx.x] = acc;
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

    for (int i = 0; i < 1; i++)
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

    for (int i = 0; i < 1; i++)
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

        for (int i = 0; i < 1; i++)
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