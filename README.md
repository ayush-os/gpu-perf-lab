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