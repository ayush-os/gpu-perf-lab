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