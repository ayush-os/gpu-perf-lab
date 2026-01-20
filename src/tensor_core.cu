#include <mma.h>
#include <cuda_fp16.h>
#include "../include/timer.cuh"
using namespace nvcuda;

// Naive matmul (CUDA cores)
__global__ void naive_matmul(float *A, float *B, float *C, int M, int N, int K)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0.f;
    for (int i = 0; i < K; i++)
        acc += A[x_idx * K + i] * B[i * K + y_idx];
    C[x_idx * N + y_idx] = acc;
}

// Tensor Core matmul using WMMA (Warp Matrix Multiply-Accumulate)
__global__ void tensor_core_matmul(half *A, half *B, float *C, int M, int N, int K)
{
    // TODO: Implement using wmma::fragment and wmma::mma_sync
    //
    // This is complex! Steps:
    // 1. Declare fragments for A, B, C
    // 2. Load tiles into fragments
    // 3. Call mma_sync to do 16x16x16 matrix multiply
    // 4. Store result
    //
    // Read: CUDA Programming Guide section on WMMA

    [REDACTED - implement tensor core version]
}

void benchmark_latency(size_t size_bytes, int stride, const char *label)
{
    const int WARMUP_ITERS = 5;
    const int TIMING_ITERS = 50;
    const int BLOCK_SIZE = 1;
    const int NUM_BLOCKS = 1;
    const int CHASE_COUNT = 100000;
    const int UNROLL_FACTOR = 10;

    int *d_output;
    cudaMalloc(&d_output, NUM_BLOCKS * BLOCK_SIZE * sizeof(int));

    int *h_chain = nullptr;
    int *d_chain = nullptr;
    prepare_pointer_chase(&h_chain, &d_chain, size_bytes, stride);

    // warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        pointer_chase_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_chain, d_output, CHASE_COUNT);
    }
    cudaDeviceSynchronize();

    // Timing runs
    CUDATimer timer;
    std::vector<float> times;

    for (int i = 0; i < TIMING_ITERS; i++)
    {
        timer.start_timer();
        pointer_chase_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_chain, d_output, CHASE_COUNT);
        float ms = timer.stop_timer();
        times.push_back(ms * 1e6f); // ms to ns
    }

    auto stats = BenchmarkStats::compute(times);

    double total_loads = (double)CHASE_COUNT * UNROLL_FACTOR;
    double latency_ns = stats.median / total_loads;

    printf("%-20s | Size: %7zu KB | Stride: %3d | Latency: %6.2f ns\n",
           label, size_bytes / 1024, stride, latency_ns);

    free(h_chain);
    cudaFree(d_chain);
    cudaFree(d_output);
}

int main()
{
    printf("=== GPU Memory Latency Benchmark (A100) ===\n");
    printf("%-20s | %-12s | %-10s | %-12s\n", "Level", "Array Size", "Stride", "Latency");
    printf("----------------------------------------------------------------------\n");

    // L1 cache: ~192 KB on A100
    // 64 KB to hit L1
    benchmark_latency(64 * 1024, 1, "L1 Cache");
    benchmark_latency(64 * 1024, 16, "L1 Cache");
    benchmark_latency(64 * 1024, 33, "L1 Cache");
    benchmark_latency(64 * 1024, 64, "L1 Cache");
    benchmark_latency(64 * 1024, 256, "L1 Cache");

    // L2 cache: 40 MB on A100
    // 4 MB to hit L2
    benchmark_latency(4 * 1024 * 1024, 1, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 16, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 33, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 64, "L2 Cache");
    benchmark_latency(4 * 1024 * 1024, 256, "L2 Cache");

    // HBM: 40/80 GB on A100
    // 512MB+ to hit HBM
    benchmark_latency(1000 * 1024 * 1024, 1, "HBM");
    benchmark_latency(1000 * 1024 * 1024, 16, "HBM");
    benchmark_latency(512 * 1024 * 1024, 33, "HBM");
    benchmark_latency(1000 * 1024 * 1024, 64, "HBM");
    benchmark_latency(1000 * 1024 * 1024, 256, "HBM");

    return 0;
}
