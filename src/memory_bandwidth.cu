#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/timer.cuh"

// --- KERNELS ---

// Optimized for L1/L2: Loops internally to stay resident in cache
__global__ void cache_benchmark_kernel(const float4* __restrict__ input, 
                                      float4* __restrict__ output, 
                                      size_t N_float4, 
                                      int inner_iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_float4) return;

    float4 val = input[idx];

    #pragma unroll 16
    for (int j = 0; j < inner_iters; j++)
    {
        // Simple math + ASM to force L1/L2 hits and prevent optimization
        val.x = val.x * 1.00001f + 0.00001f;
        asm volatile("" : "+f"(val.x), "+f"(val.y), "+f"(val.z), "+f"(val.w));
    }

    output[idx] = val;
}

// Optimized for HBM: Massive grid-stride streaming
__global__ void hbm_benchmark_kernel(const float4* __restrict__ input, 
                                    float4* __restrict__ output, 
                                    size_t N_float4)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < N_float4; i += stride)
    {
        float4 v = input[i];
        asm volatile("" : "+f"(v.x), "+f"(v.y), "+f"(v.z), "+f"(v.w));
        v.x = v.x * 3.0f + 5.0f;
        output[i] = v;
    }
}

// --- BENCHMARK RUNNER ---

void benchmark(size_t size_bytes, const char *label, int inner_iters = 1)
{
    const int BLOCK_SIZE = 256;
    // For HBM we want a huge grid; for L1/L2 we want enough to fill SMs
    int num_blocks = (label[0] == 'H') ? 16384 : 2160; 

    size_t N_float4 = size_bytes / sizeof(float4);
    float4 *d_input, *d_output;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);
    cudaMemset(d_input, 1, size_bytes);

    CUDATimer timer;
    std::vector<float> times;

    // Warmup
    if (inner_iters > 1)
        cache_benchmark_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4, 10);
    else
        hbm_benchmark_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4);
    
    cudaDeviceSynchronize();

    // Actual Measurement
    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        if (inner_iters > 1)
            cache_benchmark_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4, inner_iters);
        else
            hbm_benchmark_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4);
        
        float ms = timer.stop_timer();
        times.push_back(ms / 1000.0f);
    }

    auto stats = BenchmarkStats::compute(times);

    // Math: Account for the inner loop iterations for Cache tests
    double bytes_transferred = (double)size_bytes * 2.0 * inner_iters;
    double bandwidth_gbps = (bytes_transferred / stats.median) / 1e9;

    printf("%-15s: %10.2f GB/s\n", label, bandwidth_gbps);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    printf("=== A100 Memory Hierarchy Benchmark ===\n\n");

    // L1: Tiny data, huge iterations (Expected: ~19,000 GB/s)
    benchmark(128 * 1024, "L1 Cache", 10000);

    // L2: Fits in 40MB/80MB, medium iterations (Expected: ~5,000 GB/s)
    benchmark(32 * 1024 * 1024, "L2 Cache", 1000);

    // HBM: Massive data, 1 iteration (Expected: ~1,500-2,000 GB/s)
    benchmark(8ULL * 1024 * 1024 * 1024, "HBM (8 GB)", 1);

    return 0;
}