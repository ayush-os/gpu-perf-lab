#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include "../include/timer.cuh"

using namespace nvcuda;

// Constants for WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// --- KERNELS ---

__global__ void naive_matmul(float *A, float *B, float *C, int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float acc = 0.0f;
        for (int i = 0; i < K; i++)
        {
            acc += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] += acc;
    }
}

__global__ void tensor_core_matmul(half *A, half *B, float *C, int M, int N, int K)
{
    // TODO: Implement the WMMA logic here
    // 1. Calculate the row and column index for the WARP
    // 2. Declare fragments (wmma::fragment<wmma::matrix_a, ...>)
    // 3. Loop over the K-dimension in steps of WMMA_K
    // 4. Load, Sync, and Store
}

// --- HOST UTILITIES ---

void verify_result(float *host_C, float *device_C, int M, int N)
{
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        max_error = fmax(max_error, fabs(host_C[i] - device_C[i]));
    }
    printf("Max Error: %f\n", max_error);
    if (max_error > 0.5f)
    { // FP16 carries less precision, but shouldn't be wild
        printf("Verification FAILED!\n");
    }
    else
    {
        printf("Verification PASSED!\n");
    }
}

void cpu_matmul(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    // Problem size (Must be multiples of 16 for basic WMMA)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    printf("Matrix Dimensions: %d x %d x %d\n\n", M, N, K);

    // Allocations
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C_ref = new float[M * N];
    float *h_C_device = new float[M * N];

    half *d_A_half;
    half *d_B_half;
    float *d_A_float, *d_B_float, *d_C;

    cudaMalloc(&d_A_half, M * K * sizeof(half));
    cudaMalloc(&d_B_half, K * N * sizeof(half));
    cudaMalloc(&d_A_float, M * K * sizeof(float));
    cudaMalloc(&d_B_float, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Initialize
    for (int i = 0; i < M * K; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Copy to device (converting to half for Tensor Core)
    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    for (int i = 0; i < M * K; i++)
        h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++)
        h_B_half[i] = __float2half(h_B[i]);

    cudaMemcpy(d_A_float, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_half, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

    // --- Benchmark Naive ---
    CUDATimer timer;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    timer.start_timer();
    naive_matmul<<<grid, block>>>(d_A_float, d_B_float, d_C, M, N, K);
    float naive_ms = timer.stop_timer();

    double ops = 2.0 * M * N * K;
    printf("Naive CUDA Cores: %.2f ms (%.2f GFLOPS)\n", naive_ms, (ops * 1e-9) / (naive_ms * 1e-3));

    // --- Benchmark Tensor Core ---
    // Note: Grid/Block for WMMA is different. One Warp (32 threads) handles a 16x16 tile.
    dim3 tc_block(32, 4); // 4 warps per block
    dim3 tc_grid(N / 16, M / (16 * 4));

    cudaMemset(d_C, 0, M * N * sizeof(float));
    timer.start_timer();
    tensor_core_matmul<<<tc_grid, tc_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    float tc_ms = timer.stop_timer();

    printf("Tensor Cores WMMA: %.2f ms (%.2f TFLOPS)\n", tc_ms, (ops * 1e-12) / (tc_ms * 1e-3));

    // --- Verification ---
    cudaMemcpy(h_C_device, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);
    verify_result(h_C_ref, h_C_device, M, N);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_ref;
    delete[] h_C_device;
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C);

    return 0;
}
