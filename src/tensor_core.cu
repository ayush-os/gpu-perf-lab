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

// because the cores are loading from global memory, this achieves 90% memory utilization but only 15% compute util
__global__ void tensor_core_matmul_v1(half *A, half *B, float *C, int M, int N, int K)
{
    int warpId = threadIdx.y;                       // out of the 4 warps in my block, which warp am I a part of?
    int warpM = (blockIdx.y * blockDim.y + warpId); // get index of first warp in the block by doing blockIdx.y * blockDim.y and offset with the warpId to get exact pos for this
    int warpN = blockIdx.x;                         // the horizontal index of this warp is just blockIdx.x because we only have 1 warp horizontally in each block

    int row = warpM * WMMA_M; // need to consider the fact that each warp is doing 16 both horizontally and vertically
    int col = warpN * WMMA_N;

    if (row < M && col < N)
    {
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        // Initialize the output to zero
        wmma::fill_fragment(c_frag, 0.0f);

        for (int i = 0; i < K; i += WMMA_K)
        {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + (row * K) + i, K);
            wmma::load_matrix_sync(b_frag, B + (i * N) + col, N);

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store the output
        wmma::store_matrix_sync(C + (N * row) + col, c_frag, N, wmma::mem_row_major);
    }
}

__global__ void tensor_core_matmul_v2(half *A, half *B, float *C, int M, int N, int K)
{
    // shared memory tiling

    __shared__ half A_tile[32][16 + 8];
    __shared__ half B_tile[16][32 + 8];

    uint4 *A_vec = reinterpret_cast<uint4 *>(A_tile);
    uint4 *B_vec = reinterpret_cast<uint4 *>(B_tile);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int NUM_THREADS_IN_BLOCK = blockDim.x * blockDim.y;
    int warpId = threadIdx.y;

    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;

    int warp_row_offset = (warpId / 2) * 16;
    int warp_col_offset = (warpId % 2) * 16;

    int row_C = block_row + warp_row_offset;
    int col_C = block_col + warp_col_offset;

    for (int k_step = 0; k_step < K; k_step += WMMA_K)
    {
        if (tid < 64)
        {
            int row_tile = tid / 2; // 0 to 31
            int vec_idx = tid % 2;  // 0 or 1 (which uint4 in the row)
            int col_tile = vec_idx * 8;

            const half *gmem_ptr = &A[((blockIdx.y * 32) + row_tile) * K + (k_step + col_tile)];

            uint4 *row_ptr = reinterpret_cast<uint4 *>(&A_tile[row_tile][0]);
            row_ptr[vec_idx] = *reinterpret_cast<const uint4 *>(gmem_ptr);
        }
        else
        {
            int tid_b = tid - 64;     // 0 to 63
            int row_tile = tid_b / 4; // 0 to 15
            int vec_idx = tid_b % 4;  // 0, 1, 2, or 3
            int col_tile = vec_idx * 8;

            const half *gmem_ptr = &B[(k_step + row_tile) * N + ((blockIdx.x * 32) + col_tile)];

            uint4 *row_ptr = reinterpret_cast<uint4 *>(&B_tile[row_tile][0]);
            row_ptr[vec_idx] = *reinterpret_cast<const uint4 *>(gmem_ptr);
        }

        __syncthreads();

        int a_tile_row = (warpId / 2) * WMMA_K;
        int b_tile_col = (warpId % 2) * WMMA_K;
        wmma::load_matrix_sync(a_frag, &A_tile[a_tile_row][0], 16 + 8);
        wmma::load_matrix_sync(b_frag, &B_tile[0][b_tile_col], 32 + 8);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
}

__global__ void tensor_core_matmul_v3(half *A, half *B, float *C, int M, int N, int K)
{
    // Larger K-tile: 64 instead of 16
    // This means 4 WMMA operations per tile load, amortizing global memory cost
    const int K_TILE = 64;

    // Shared memory: 32 rows x 64 cols for A, 64 rows x 32 cols for B
    // Padding to avoid bank conflicts
    __shared__ half A_tile[32][K_TILE + 8];
    __shared__ half B_tile[K_TILE][32 + 8];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = threadIdx.y;

    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;

    int warp_row_offset = (warpId / 2) * 16;
    int warp_col_offset = (warpId % 2) * 16;

    int row_C = block_row + warp_row_offset;
    int col_C = block_col + warp_col_offset;

    int a_tile_row = (warpId / 2) * WMMA_M;
    int b_tile_col = (warpId % 2) * WMMA_N;

    // We have 128 threads (4 warps * 32 threads)
    // A_tile: 32 rows x 64 cols = 2048 halfs = 256 uint4s (8 halfs per uint4)
    // B_tile: 64 rows x 32 cols = 2048 halfs = 256 uint4s
    // Total: 512 uint4 loads, 128 threads -> 4 loads per thread

    for (int k_step = 0; k_step < K; k_step += K_TILE)
    {
// Load A_tile: 32 x 64 = 2048 elements = 256 uint4s
// Each thread loads 2 uint4s for A
#pragma unroll
        for (int load = 0; load < 2; load++)
        {
            int linear_idx = tid + load * 128; // 0-255
            int row_tile = linear_idx / 8;     // 0-31 (64 cols / 8 = 8 uint4s per row)
            int vec_idx = linear_idx % 8;      // 0-7
            int col_tile = vec_idx * 8;

            const half *gmem_ptr = &A[(block_row + row_tile) * K + (k_step + col_tile)];
            uint4 *row_ptr = reinterpret_cast<uint4 *>(&A_tile[row_tile][0]);
            row_ptr[vec_idx] = *reinterpret_cast<const uint4 *>(gmem_ptr);
        }

// Load B_tile: 64 x 32 = 2048 elements = 256 uint4s
// Each thread loads 2 uint4s for B
#pragma unroll
        for (int load = 0; load < 2; load++)
        {
            int linear_idx = tid + load * 128; // 0-255
            int row_tile = linear_idx / 4;     // 0-63 (32 cols / 8 = 4 uint4s per row)
            int vec_idx = linear_idx % 4;      // 0-3
            int col_tile = vec_idx * 8;

            const half *gmem_ptr = &B[(k_step + row_tile) * N + (block_col + col_tile)];
            uint4 *row_ptr = reinterpret_cast<uint4 *>(&B_tile[row_tile][0]);
            row_ptr[vec_idx] = *reinterpret_cast<const uint4 *>(gmem_ptr);
        }

        __syncthreads();

// Now do 4 WMMA operations along the K dimension
#pragma unroll
        for (int k_inner = 0; k_inner < K_TILE; k_inner += WMMA_K)
        {
            wmma::load_matrix_sync(a_frag[0], &A_tile[a_tile_row][k_inner], K_TILE + 8);
            wmma::load_matrix_sync(b_frag[0], &B_tile[k_inner][b_tile_col], 32 + 8);
            wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
        }

        __syncthreads();
    }

    wmma::store_matrix_sync(C + (N * row_C) + col_C, c_frag, N, wmma::mem_row_major);
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

    // --- Benchmark Tensor Core V1 ---
    dim3 tc_block(32, 4); // 4 warps per block
    dim3 tc_grid(N / 16, M / (16 * 4));

    cudaMemset(d_C, 0, M * N * sizeof(float));
    timer.start_timer();
    tensor_core_matmul_v1<<<tc_grid, tc_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    float tc_ms = timer.stop_timer();

    printf("Tensor Cores WMMA: %.2f ms (%.2f TFLOPS)\n", tc_ms, (ops * 1e-12) / (tc_ms * 1e-3));

    // --- Verification ---
    // cudaMemcpy(h_C_device, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cpu_matmul(h_A, h_B, h_C_ref, M, N, K);
    // verify_result(h_C_ref, h_C_device, M, N);

    // --- Benchmark Tensor Core V2 ---

    dim3 tc2_block(32, 4); // 4 warps per block
    dim3 tc2_grid(N / 32, M / 32);

    cudaMemset(d_C, 0, M * N * sizeof(float));
    timer.start_timer();
    tensor_core_matmul_v2<<<tc2_grid, tc2_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    float tc2_ms = timer.stop_timer();

    printf("Tensor Cores Smem WMMA: %.2f ms (%.2f TFLOPS)\n", tc2_ms, (ops * 1e-12) / (tc2_ms * 1e-3));

    // --- Verification ---
    // cudaMemcpy(h_C_device, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cpu_matmul(h_A, h_B, h_C_ref, M, N, K);
    // verify_result(h_C_ref, h_C_device, M, N);

    dim3 tc3_block(32, 4); // 4 warps per block
    dim3 tc3_grid(N / 32, M / 32);

    cudaMemset(d_C, 0, M * N * sizeof(float));
    timer.start_timer();
    tensor_core_matmul_v3<<<tc3_grid, tc3_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    float tc3_ms = timer.stop_timer();

    printf("Tensor Cores K64 WMMA: %.2f ms (%.2f TFLOPS)\n", tc3_ms, (ops * 1e-12) / (tc3_ms * 1e-3));

    // --- Verification ---
    cudaMemcpy(h_C_device, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
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
