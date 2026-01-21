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
    output[blockIdx.x * blockDim.x + threadIdx.x] = val;
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

    float val = smem[threadIdx.x * stride];
    output[blockIdx.x * blockDim.x + threadIdx.x] = val;
}

// Broadcast (special case - no conflict)
__global__ void broadcast_access(float *output, int N)
{
    __shared__ float smem[256];

    smem[threadIdx.x] = threadIdx.x * 1.0f;
    __syncthreads();

    float val = smem[0];
    output[blockIdx.x * blockDim.x + threadIdx.x] = val;
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