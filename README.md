# gpu-perf-lab

### part one: register bandwidth kernel to measure peak compute
Goal: achieve max GFLOPs on the GPU when compute bound
Aka all memory is in registers

First kernel: Very low ILP, FMA, only 1 block. GFLOPS: 12.51
__global__ void register_bandwidth_test(float *output, int iterations)
{
    float acc = 0.1f;
    for (int i = 0; i < iterations; i++)
    {
        acc += 1.01 * acc;
    }

    output[threadIdx.x] = acc;
}

Second Kernel: Saturate SMs with 432 blocks. immediate jump 1762.73

Third Kernel: Add ILP to prev. Messed up metrics on this one but i think it jumped to 14,726.96

Switched to a different kernel: 7271.75 GFLOPs
__global__ void register_bandwidth_test(float *out, int iterations)
{
    float acc = 0.0f;
    for (int i = 0; i < iterations; i++)
    {
        acc += 1.0f;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

ILP optimized new kernel: 7764.06 GFLOP by having 4 accums then 7821.34 via 8 accum

Then to 15406.00 GFLOPs via FFMA instruction (doing mult and add in 1 cycle) - from here on out i think they are all similar perf (max utilization)

then to 15611.86 GFLOPs by adding a loop unroll #pragma unroll 32

then to PTX i get 15604.67+

then go from 8 accums to 12 accums for even more ILP

then go from block size 432 to 864 for even more saturation

### part two: pointer chasing to determine memory latency
Goal: expose memory access latency for each part of the memory hierarchy

Using pointer chasing trick by making every subsequent memory access dependent on the result found in the previous one
Prevents the memory prefetcher from recognizing the pattern and prefetching into L1

Need to control for stride because in the case of stride = 1 the prefetcher isn't fooled. But as soon as you go over the cache line size (128 bytes) with a stride of 33 and above, you see the discrepancy because the load is happening from L1, then L2, then HBM - aka successfully fooled the prefetcher/caching mechanisms. Stride 16 also seems to be complex enough to fool the prefetcher

registers -> L1 -> L2 -> HBM
clock cycle -> SM proximity -> crossbar/partition -> DRAM physics

### part three: trying to saturate memory bandwidth across HBM, L1, L2

HBM was easy - i was able to use a standard streaming kernel with float4 arrays and a accumulator to achieve 1.7+ (84% utilization) of the A100 SXM4's 2 TB/s HBM bandwidth

L1 and L2 were a fail. L2 kept on coalescing my writes and so i was getting extremely low bandwidth usage and for the L1 kernel i just had 1 float that kept getting added to, so then it ended up just staying in a register and writing back to L1 once after the 10000 iterations instead of every single time, again leading to extremely low bandwidth. :(

### part four: memory access patterns

results are astounding - randomization causes a 34x slowdown compared to perfect coalescing. makes sense why increases in stride cause a proportional increase but then the slowdowns taper off because once each element is already on a separate cache line, further strides do not have the same doubling effect.

N: 64M elements (256MB per array)

Coalesced              | Time:   359.42 us | BW: 1493.70 GB/s | Slowdown:  1.00x
Strided      (stride  2) | Time:   902.14 us | BW:  595.11 GB/s | Slowdown:  2.51x
Strided      (stride  4) | Time:  1773.57 us | BW:  302.71 GB/s | Slowdown:  4.93x
Strided      (stride  8) | Time:  3538.94 us | BW:  151.70 GB/s | Slowdown:  9.85x
Strided      (stride 16) | Time:  4818.94 us | BW:  111.41 GB/s | Slowdown: 13.41x
Strided      (stride 32) | Time:  5284.86 us | BW:  101.59 GB/s | Slowdown: 14.70x
Strided      (stride 64) | Time:  6879.23 us | BW:   78.04 GB/s | Slowdown: 19.14x
Randomized             | Time: 12366.85 us | BW:   43.41 GB/s | Slowdown: 34.41x

### part five: warp divergence

again, just a simple experiemnt but shows the importance of avoiding conditions that split up threads in a warp. slowdown is directly proportional to the divergence factor - 2x divergence factor causes a 2x slowdown.

no_divergence          | Time:  6952.96 us | Slowdown:  1.00x
full_divergence        | Time: 269455.88 us | Slowdown: 38.75x
partial_divergence (divergence_factor  2) | Time: 13942.78 us | Slowdown:  2.01x
partial_divergence (divergence_factor  4) | Time: 27761.66 us | Slowdown:  3.99x
partial_divergence (divergence_factor  8) | Time: 55406.59 us | Slowdown:  7.97x
partial_divergence (divergence_factor 16) | Time: 121791.49 us | Slowdown: 17.52x
partial_divergence (divergence_factor 32) | Time: 269912.09 us | Slowdown: 38.82x

### part six: tensor cores - holy crap

Matrix Dimensions: 1024 x 1024 x 1024

Naive CUDA Cores: 5.49 ms (391.05 GFLOPS)
Tensor Cores WMMA: 0.15 ms (14.07 TFLOPS) ---> 36x!!!
Max Error: 0.016907
Verification PASSED!

but still extremely memory bound lol
Memory Throughput (89.90%)
Compute Throughput (15.86%)
Eligible Warps (percycle i think?) (0.13)

Switching from ALU to Tensor cores did significantly increase arithmetic intensity because tensor cores operated on tiles of 16x16 instead of ALU which was essentially doing 1x1.
But the kernel is still incredibly memory bound and tensor cores are idle waiting for memory that we are reading in and storing to HBM every single time. The much better way to do this is to have a 
shared memory buffer that we would use to dramatically lower gmmem reads/writes.

### part seven: shared memory

=== Shared Memory Bank Conflicts ===

No Conflicts    | Time:    20.48 us | BW:  204.80 GB/s | Slowdown:  1.00x
Broadcast       | Time:    20.48 us | BW:  204.80 GB/s | Slowdown:  1.00x

--- Testing Strided Access (Bank Conflicts) ---
Stride 1        | Time:    36.86 us | BW:  113.78 GB/s | Slowdown:  1.80x
Stride 2        | Time:    39.94 us | BW:  105.03 GB/s | Slowdown:  1.95x
Stride 4        | Time:    54.27 us | BW:   77.28 GB/s | Slowdown:  2.65x
Stride 8        | Time:    83.97 us | BW:   49.95 GB/s | Slowdown:  4.10x
Stride 16       | Time:   143.36 us | BW:   29.26 GB/s | Slowdown:  7.00x
Stride 32       | Time:   262.14 us | BW:   16.00 GB/s | Slowdown: 12.80x (32-way conflict!)
Stride 33       | Time:    36.86 us | BW:  113.78 GB/s | Slowdown:  1.80x (Conflict-free)
Stride 64       | Time:   262.14 us | BW:   16.00 GB/s | Slowdown: 12.80x

Hard to trick the compiler here but was able to get a pretty decent sense of how overlapping accesses to the same bank in smem causes a slowdown

stride = 1 had
 l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  7,573 (prolly due to some internal bookkeeping from calling __synchtreads() 131000 times)

 stride = 32 had 
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum 32,517,775

no_conflicts and broadcasting had
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0

Holy!!!! 4000x increase in bank conflicts from stride = 1 to stride = 32
Slowdown was not as bad because gpu was still doing other useful work during the serialization of the reads from smem.


### part eight: optimizing tensor core matmul with smem
paperspace@ps83msra5sg7:~/gpu-perf-lab$ sudo $(which ncu) --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum --kernel-name tensor_core_matmul_v2 ./build/tensor_core
Matrix Dimensions: 1024 x 1024 x 1024

==PROF== Connected to process 3171 (/home/paperspace/gpu-perf-lab/build/tensor_core)
Naive CUDA Cores: 5.50 ms (390.43 GFLOPS)
Tensor Cores WMMA: 0.41 ms (5.19 TFLOPS)
Max Error: 0.016907
Verification PASSED!
==PROF== Profiling "tensor_core_matmul_v2": 0%....50%....100% - 1 pass
Tensor Cores Smem WMMA: 379.80 ms (0.01 TFLOPS)
Max Error: 0.016907
Verification PASSED!
==PROF== Disconnected from process 3171
==WARNING== Found outstanding GPU clock reset, trying to revert...Success.
[3171] tensor_core@127.0.0.1
  tensor_core_matmul_v2(__half *, __half *, float *, int, int, int) (32, 32, 1)x(32, 4, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                4,201,261
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                    6,320,726
    -------------------------------------------------------- ----------- ------------

paperspace@ps83msra5sg7:~/gpu-perf-lab$ ./build/tensor_core 
Matrix Dimensions: 4096 x 4096 x 4096

Naive CUDA Cores: 277.74 ms (494.85 GFLOPS)
Tensor Cores WMMA: 7.74 ms (17.77 TFLOPS)
Tensor Cores Smem WMMA: 4.02 ms (34.20 TFLOPS)