#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>

// --- KERNELS ---

// L1: Stays in registers/L1.
// We multiply by inner_iters because we are measuring throughput of L1 access.
__global__ void l1_benchmark_kernel(float4 *data, int inner_iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = data[idx];

#pragma unroll 16
    for (int j = 0; j < inner_iters; j++)
    {
        val.x = val.x * 1.00001f + 0.00001f;
        // Use ASM to prevent the compiler from optimizing the loop away
        asm volatile("" : "+f"(val.x), "+f"(val.y), "+f"(val.z), "+f"(val.w));
    }
    data[idx] = val;
}

// L2: Bypasses L1 using .cg (cache global) modifier
// This forces every load/store to hit L2.
__global__ void l2_benchmark_resident_kernel(float4 *data, size_t N, int inner_iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float4 val;
    // We use one pointer to keep the working set small and inside L2
    for (int j = 0; j < inner_iters; j++)
    {
        // Force L2 access (bypass L1)
        asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(&data[idx]));

        val.x += 0.00001f; // Minimal math

        asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
                     : : "l"(&data[idx]), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
    }
}

// HBM: Standard streaming kernel
__global__ void hbm_benchmark_kernel(const float4 *__restrict__ input,
                                     float4 *__restrict__ output,
                                     size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride)
    {
        float4 v = input[i];
        v.x = v.x * 1.00001f + 0.00001f;
        output[i] = v;
    }
}

// --- RUNNER ---

void run_benchmark(const char *mode, size_t size_bytes, int inner_iters = 1)
{
    float4 *d_input, *d_output;
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, size_bytes);

    size_t N = size_bytes / sizeof(float4);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535)
        blocks = 65535; // Cap for HBM stride

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    hbm_benchmark_kernel<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 50; i++)
    {
        if (strcmp(mode, "L1") == 0)
            l1_benchmark_kernel<<<blocks, threads>>>(d_input, inner_iters);
        else if (strcmp(mode, "L2") == 0)
            l2_benchmark_resident_kernel<<<blocks, threads>>>(d_input, N, inner_iters);
        else
            hbm_benchmark_kernel<<<blocks, threads>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_s = (ms / 50.0) / 1000.0;

    // Bandwidth Logic:
    // L1: data_size * 2 (load/store) * iters
    // L2/HBM: data_size * 2 (load/store)
    double bytes;
    if (strcmp(mode, "L1") == 0 || strcmp(mode, "L2") == 0)
    {
        bytes = (double)size_bytes * 2 * inner_iters;
    }
    else
    {
        bytes = (double)size_bytes * 2;
    }
    double bw = (bytes / avg_s) / 1e9;

    printf("%-10s : %10.2f GB/s (Size: %zu MB)\n", mode, bw, size_bytes / (1024 * 1024));

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    // L1: 128KB (Fits in SM L1)
    run_benchmark("L1", 128 * 1024, 10000);

    // L2: 32MB (Fits in A100 40MB/80MB L2)
    run_benchmark("L2", 20 * 1024 * 1024, 100);

    // HBM: 4GB
    run_benchmark("HBM", 4ULL * 1024 * 1024 * 1024);

    return 0;
}