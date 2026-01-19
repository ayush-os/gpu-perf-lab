#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include "../include/timer.cuh"

// --- KERNELS ---

// L1: Measures Register/L1 throughput by looping on values already in registers.
__global__ void l1_kernel(const float4 *__restrict__ input, float4 *__restrict__ output, int inner_iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = input[idx];

    for (int j = 0; j < inner_iters; j++)
    {
        val.x = val.x * 1.00001f + 0.00001f;
        // ASM "clobber" ensures the loop isn't optimized away, but keeps it in registers
        asm volatile("" : "+f"(val.x), "+f"(val.y), "+f"(val.z), "+f"(val.w));
    }
    output[idx] = val;
}

// L2: Forces a trip to the L2 Cache silicon by using threadfences and memory clobbers.
__global__ void l2_kernel(const float4 *__restrict__ input, float4 *__restrict__ output, size_t N_float4, int inner_iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_float4)
        return;

    for (int j = 0; j < inner_iters; j++)
    {
        float4 val = input[idx];
        val.x = val.x * 1.00001f + 0.00001f;

        // Force the write to L2 and ensure it's visible (prevents register caching)
        output[idx] = val;
        __threadfence();

        // Ensure the compiler doesn't reorder or skip these cycles
        asm volatile("" : : "f"(val.x) : "memory");
    }
}

// HBM: Standard grid-stride streaming to saturate the HBM2e controllers.
__global__ void hbm_kernel(const float4 *__restrict__ input, float4 *__restrict__ output, size_t N_float4)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < N_float4; i += stride)
    {
        float4 v = input[i];
        v.x = v.x * 3.0f + 5.0f;
        output[i] = v;
    }
}

// --- BENCHMARK RUNNER ---

void benchmark_tier(size_t size_bytes, const char *label, int type, int inner_iters = 1)
{
    const int BLOCK_SIZE = 256;
    int num_blocks = (type == 2) ? 16384 : 2160; // Huge grid for HBM, enough to saturate SMs for Cache

    size_t N_float4 = size_bytes / sizeof(float4);
    float4 *d_input, *d_output;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);
    cudaMemset(d_input, 1, size_bytes);

    CUDATimer timer;
    std::vector<float> times;

    // Warmup
    if (type == 0)
        l1_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, 10);
    else if (type == 1)
        l2_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4, 10);
    else
        hbm_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        if (type == 0)
            l1_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, inner_iters);
        else if (type == 1)
            l2_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4, inner_iters);
        else
            hbm_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N_float4);
        times.push_back(timer.stop_timer() / 1000.0f);
    }

    auto stats = BenchmarkStats::compute(times);
    double total_bytes = (double)size_bytes * 2.0 * inner_iters;
    double bandwidth_gbps = (total_bytes / stats.median) / 1e9;

    printf("%-15s: %10.2f GB/s\n", label, bandwidth_gbps);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    printf("=== A100 Final Optimized Memory Benchmark ===\n\n");

    // L1: 128 KB, huge internal loop (Type 0)
    benchmark_tier(128 * 1024, "L1 Cache", 0, 10000);

    // L2: 32 MB, medium loop + fences (Type 1)
    // 32MB fits in A100 L2 but exceeds L1
    benchmark_tier(32 * 1024 * 1024, "L2 Cache", 1, 100);

    // HBM: 8 GB, streaming (Type 2)
    benchmark_tier(8ULL * 1024 * 1024 * 1024, "HBM (8 GB)", 2, 1);

    return 0;
}