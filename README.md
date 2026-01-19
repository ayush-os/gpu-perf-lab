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

