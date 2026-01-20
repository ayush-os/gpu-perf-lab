#include <stdio.h>
#include "../include/timer.cuh"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstring>

// Low register pressure (simple kernel)
__global__ void low_registers(float *data, int N)
{
    // Uses 8 registers
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = data[idx];
        x = x * 2.0f + 1.0f;
        data[idx] = x;
    }
}

// uses 136 regs
__global__ void high_registers(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = data[idx];

        // ============================================================
        // PHASE 1: Create 128 variables with sequential dependencies
        // ============================================================
        float v0 = x, v1, v2, v3, v4, v5, v6, v7, v8, v9;
        float v10, v11, v12, v13, v14, v15, v16, v17, v18, v19;
        float v20, v21, v22, v23, v24, v25, v26, v27, v28, v29;
        float v30, v31, v32, v33, v34, v35, v36, v37, v38, v39;
        float v40, v41, v42, v43, v44, v45, v46, v47, v48, v49;
        float v50, v51, v52, v53, v54, v55, v56, v57, v58, v59;
        float v60, v61, v62, v63, v64, v65, v66, v67, v68, v69;
        float v70, v71, v72, v73, v74, v75, v76, v77, v78, v79;
        float v80, v81, v82, v83, v84, v85, v86, v87, v88, v89;
        float v90, v91, v92, v93, v94, v95, v96, v97, v98, v99;
        float v100, v101, v102, v103, v104, v105, v106, v107, v108, v109;
        float v110, v111, v112, v113, v114, v115, v116, v117, v118, v119;
        float v120, v121, v122, v123, v124, v125, v126, v127;

        // Sequential chain - each depends on previous
        v1 = v0 * 1.001f + 0.1f;
        v2 = v1 * 1.001f + 0.1f;
        v3 = v2 * 1.001f + 0.1f;
        v4 = v3 * 1.001f + 0.1f;
        v5 = v4 * 1.001f + 0.1f;
        v6 = v5 * 1.001f + 0.1f;
        v7 = v6 * 1.001f + 0.1f;
        v8 = v7 * 1.001f + 0.1f;
        v9 = v8 * 1.001f + 0.1f;
        v10 = v9 * 1.001f + 0.1f;
        v11 = v10 * 1.001f + 0.1f;
        v12 = v11 * 1.001f + 0.1f;
        v13 = v12 * 1.001f + 0.1f;
        v14 = v13 * 1.001f + 0.1f;
        v15 = v14 * 1.001f + 0.1f;
        v16 = v15 * 1.001f + 0.1f;
        v17 = v16 * 1.001f + 0.1f;
        v18 = v17 * 1.001f + 0.1f;
        v19 = v18 * 1.001f + 0.1f;
        v20 = v19 * 1.001f + 0.1f;
        v21 = v20 * 1.001f + 0.1f;
        v22 = v21 * 1.001f + 0.1f;
        v23 = v22 * 1.001f + 0.1f;
        v24 = v23 * 1.001f + 0.1f;
        v25 = v24 * 1.001f + 0.1f;
        v26 = v25 * 1.001f + 0.1f;
        v27 = v26 * 1.001f + 0.1f;
        v28 = v27 * 1.001f + 0.1f;
        v29 = v28 * 1.001f + 0.1f;
        v30 = v29 * 1.001f + 0.1f;
        v31 = v30 * 1.001f + 0.1f;
        v32 = v31 * 1.001f + 0.1f;
        v33 = v32 * 1.001f + 0.1f;
        v34 = v33 * 1.001f + 0.1f;
        v35 = v34 * 1.001f + 0.1f;
        v36 = v35 * 1.001f + 0.1f;
        v37 = v36 * 1.001f + 0.1f;
        v38 = v37 * 1.001f + 0.1f;
        v39 = v38 * 1.001f + 0.1f;
        v40 = v39 * 1.001f + 0.1f;
        v41 = v40 * 1.001f + 0.1f;
        v42 = v41 * 1.001f + 0.1f;
        v43 = v42 * 1.001f + 0.1f;
        v44 = v43 * 1.001f + 0.1f;
        v45 = v44 * 1.001f + 0.1f;
        v46 = v45 * 1.001f + 0.1f;
        v47 = v46 * 1.001f + 0.1f;
        v48 = v47 * 1.001f + 0.1f;
        v49 = v48 * 1.001f + 0.1f;
        v50 = v49 * 1.001f + 0.1f;
        v51 = v50 * 1.001f + 0.1f;
        v52 = v51 * 1.001f + 0.1f;
        v53 = v52 * 1.001f + 0.1f;
        v54 = v53 * 1.001f + 0.1f;
        v55 = v54 * 1.001f + 0.1f;
        v56 = v55 * 1.001f + 0.1f;
        v57 = v56 * 1.001f + 0.1f;
        v58 = v57 * 1.001f + 0.1f;
        v59 = v58 * 1.001f + 0.1f;
        v60 = v59 * 1.001f + 0.1f;
        v61 = v60 * 1.001f + 0.1f;
        v62 = v61 * 1.001f + 0.1f;
        v63 = v62 * 1.001f + 0.1f;
        v64 = v63 * 1.001f + 0.1f;
        v65 = v64 * 1.001f + 0.1f;
        v66 = v65 * 1.001f + 0.1f;
        v67 = v66 * 1.001f + 0.1f;
        v68 = v67 * 1.001f + 0.1f;
        v69 = v68 * 1.001f + 0.1f;
        v70 = v69 * 1.001f + 0.1f;
        v71 = v70 * 1.001f + 0.1f;
        v72 = v71 * 1.001f + 0.1f;
        v73 = v72 * 1.001f + 0.1f;
        v74 = v73 * 1.001f + 0.1f;
        v75 = v74 * 1.001f + 0.1f;
        v76 = v75 * 1.001f + 0.1f;
        v77 = v76 * 1.001f + 0.1f;
        v78 = v77 * 1.001f + 0.1f;
        v79 = v78 * 1.001f + 0.1f;
        v80 = v79 * 1.001f + 0.1f;
        v81 = v80 * 1.001f + 0.1f;
        v82 = v81 * 1.001f + 0.1f;
        v83 = v82 * 1.001f + 0.1f;
        v84 = v83 * 1.001f + 0.1f;
        v85 = v84 * 1.001f + 0.1f;
        v86 = v85 * 1.001f + 0.1f;
        v87 = v86 * 1.001f + 0.1f;
        v88 = v87 * 1.001f + 0.1f;
        v89 = v88 * 1.001f + 0.1f;
        v90 = v89 * 1.001f + 0.1f;
        v91 = v90 * 1.001f + 0.1f;
        v92 = v91 * 1.001f + 0.1f;
        v93 = v92 * 1.001f + 0.1f;
        v94 = v93 * 1.001f + 0.1f;
        v95 = v94 * 1.001f + 0.1f;
        v96 = v95 * 1.001f + 0.1f;
        v97 = v96 * 1.001f + 0.1f;
        v98 = v97 * 1.001f + 0.1f;
        v99 = v98 * 1.001f + 0.1f;
        v100 = v99 * 1.001f + 0.1f;
        v101 = v100 * 1.001f + 0.1f;
        v102 = v101 * 1.001f + 0.1f;
        v103 = v102 * 1.001f + 0.1f;
        v104 = v103 * 1.001f + 0.1f;
        v105 = v104 * 1.001f + 0.1f;
        v106 = v105 * 1.001f + 0.1f;
        v107 = v106 * 1.001f + 0.1f;
        v108 = v107 * 1.001f + 0.1f;
        v109 = v108 * 1.001f + 0.1f;
        v110 = v109 * 1.001f + 0.1f;
        v111 = v110 * 1.001f + 0.1f;
        v112 = v111 * 1.001f + 0.1f;
        v113 = v112 * 1.001f + 0.1f;
        v114 = v113 * 1.001f + 0.1f;
        v115 = v114 * 1.001f + 0.1f;
        v116 = v115 * 1.001f + 0.1f;
        v117 = v116 * 1.001f + 0.1f;
        v118 = v117 * 1.001f + 0.1f;
        v119 = v118 * 1.001f + 0.1f;
        v120 = v119 * 1.001f + 0.1f;
        v121 = v120 * 1.001f + 0.1f;
        v122 = v121 * 1.001f + 0.1f;
        v123 = v122 * 1.001f + 0.1f;
        v124 = v123 * 1.001f + 0.1f;
        v125 = v124 * 1.001f + 0.1f;
        v126 = v125 * 1.001f + 0.1f;
        v127 = v126 * 1.001f + 0.1f;

        // ============================================================
        // PHASE 2: Cross-link ALL variables so they're ALL live at once
        // ============================================================
        v0 += v127 * 0.001f;
        v1 += v126 * 0.001f;
        v2 += v125 * 0.001f;
        v3 += v124 * 0.001f;
        v4 += v123 * 0.001f;
        v5 += v122 * 0.001f;
        v6 += v121 * 0.001f;
        v7 += v120 * 0.001f;
        v8 += v119 * 0.001f;
        v9 += v118 * 0.001f;
        v10 += v117 * 0.001f;
        v11 += v116 * 0.001f;
        v12 += v115 * 0.001f;
        v13 += v114 * 0.001f;
        v14 += v113 * 0.001f;
        v15 += v112 * 0.001f;
        v16 += v111 * 0.001f;
        v17 += v110 * 0.001f;
        v18 += v109 * 0.001f;
        v19 += v108 * 0.001f;
        v20 += v107 * 0.001f;
        v21 += v106 * 0.001f;
        v22 += v105 * 0.001f;
        v23 += v104 * 0.001f;
        v24 += v103 * 0.001f;
        v25 += v102 * 0.001f;
        v26 += v101 * 0.001f;
        v27 += v100 * 0.001f;
        v28 += v99 * 0.001f;
        v29 += v98 * 0.001f;
        v30 += v97 * 0.001f;
        v31 += v96 * 0.001f;
        v32 += v95 * 0.001f;
        v33 += v94 * 0.001f;
        v34 += v93 * 0.001f;
        v35 += v92 * 0.001f;
        v36 += v91 * 0.001f;
        v37 += v90 * 0.001f;
        v38 += v89 * 0.001f;
        v39 += v88 * 0.001f;
        v40 += v87 * 0.001f;
        v41 += v86 * 0.001f;
        v42 += v85 * 0.001f;
        v43 += v84 * 0.001f;
        v44 += v83 * 0.001f;
        v45 += v82 * 0.001f;
        v46 += v81 * 0.001f;
        v47 += v80 * 0.001f;
        v48 += v79 * 0.001f;
        v49 += v78 * 0.001f;
        v50 += v77 * 0.001f;
        v51 += v76 * 0.001f;
        v52 += v75 * 0.001f;
        v53 += v74 * 0.001f;
        v54 += v73 * 0.001f;
        v55 += v72 * 0.001f;
        v56 += v71 * 0.001f;
        v57 += v70 * 0.001f;
        v58 += v69 * 0.001f;
        v59 += v68 * 0.001f;
        v60 += v67 * 0.001f;
        v61 += v66 * 0.001f;
        v62 += v65 * 0.001f;
        v63 += v64 * 0.001f;

        // Reverse direction cross-links
        v127 += v0 * 0.001f;
        v126 += v1 * 0.001f;
        v125 += v2 * 0.001f;
        v124 += v3 * 0.001f;
        v123 += v4 * 0.001f;
        v122 += v5 * 0.001f;
        v121 += v6 * 0.001f;
        v120 += v7 * 0.001f;
        v119 += v8 * 0.001f;
        v118 += v9 * 0.001f;
        v117 += v10 * 0.001f;
        v116 += v11 * 0.001f;
        v115 += v12 * 0.001f;
        v114 += v13 * 0.001f;
        v113 += v14 * 0.001f;
        v112 += v15 * 0.001f;
        v111 += v16 * 0.001f;
        v110 += v17 * 0.001f;
        v109 += v18 * 0.001f;
        v108 += v19 * 0.001f;
        v107 += v20 * 0.001f;
        v106 += v21 * 0.001f;
        v105 += v22 * 0.001f;
        v104 += v23 * 0.001f;
        v103 += v24 * 0.001f;
        v102 += v25 * 0.001f;
        v101 += v26 * 0.001f;
        v100 += v27 * 0.001f;
        v99 += v28 * 0.001f;
        v98 += v29 * 0.001f;
        v97 += v30 * 0.001f;
        v96 += v31 * 0.001f;
        v95 += v32 * 0.001f;
        v94 += v33 * 0.001f;
        v93 += v34 * 0.001f;
        v92 += v35 * 0.001f;
        v91 += v36 * 0.001f;
        v90 += v37 * 0.001f;
        v89 += v38 * 0.001f;
        v88 += v39 * 0.001f;
        v87 += v40 * 0.001f;
        v86 += v41 * 0.001f;
        v85 += v42 * 0.001f;
        v84 += v43 * 0.001f;
        v83 += v44 * 0.001f;
        v82 += v45 * 0.001f;
        v81 += v46 * 0.001f;
        v80 += v47 * 0.001f;
        v79 += v48 * 0.001f;
        v78 += v49 * 0.001f;
        v77 += v50 * 0.001f;
        v76 += v51 * 0.001f;
        v75 += v52 * 0.001f;
        v74 += v53 * 0.001f;
        v73 += v54 * 0.001f;
        v72 += v55 * 0.001f;
        v71 += v56 * 0.001f;
        v70 += v57 * 0.001f;
        v69 += v58 * 0.001f;
        v68 += v59 * 0.001f;
        v67 += v60 * 0.001f;
        v66 += v61 * 0.001f;
        v65 += v62 * 0.001f;
        v64 += v63 * 0.001f;

        // ============================================================
        // PHASE 3: Use ALL values (prevents dead code elimination)
        // ============================================================
        float result =
            v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 +
            v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 +
            v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27 + v28 + v29 +
            v30 + v31 + v32 + v33 + v34 + v35 + v36 + v37 + v38 + v39 +
            v40 + v41 + v42 + v43 + v44 + v45 + v46 + v47 + v48 + v49 +
            v50 + v51 + v52 + v53 + v54 + v55 + v56 + v57 + v58 + v59 +
            v60 + v61 + v62 + v63 + v64 + v65 + v66 + v67 + v68 + v69 +
            v70 + v71 + v72 + v73 + v74 + v75 + v76 + v77 + v78 + v79 +
            v80 + v81 + v82 + v83 + v84 + v85 + v86 + v87 + v88 + v89 +
            v90 + v91 + v92 + v93 + v94 + v95 + v96 + v97 + v98 + v99 +
            v100 + v101 + v102 + v103 + v104 + v105 + v106 + v107 + v108 + v109 +
            v110 + v111 + v112 + v113 + v114 + v115 + v116 + v117 + v118 + v119 +
            v120 + v121 + v122 + v123 + v124 + v125 + v126 + v127;

        data[idx] = result;
    }
}

// Result struct for storing benchmark data
struct Result
{
    const char *name;
    int block_size;
    int registers;
    int blocks_per_sm;
    int warps_per_sm;
    float occupancy;
    float time_us;
};

void run_reg_pressure_analysis()
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);
    const int NUM_RUNS = 100;
    const int WARMUP_RUNS = 10;

    // Allocate memory
    float *d_data;
    cudaMalloc(&d_data, bytes);

    // Initialize with random data
    std::vector<float> h_data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++)
    {
        h_data[i] = dist(rng);
    }
    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("SMs: %d\n\n", prop.multiProcessorCount);

    // Test different block sizes
    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};

    printf("================================================================================\n");
    printf("OCCUPANCY ANALYSIS\n");
    printf("================================================================================\n");
    printf("%-12s | %-8s | %-8s | %-10s | %-10s | %-12s\n",
           "Kernel", "BlkSize", "Regs", "Blocks/SM", "Warps/SM", "Occupancy");
    printf("--------------------------------------------------------------------------------\n");

    // Store results for analysis
    std::vector<Result> results;

    for (int block_size : block_sizes)
    {
        int grid_size = (N + block_size - 1) / block_size;

        // Low registers kernel
        {
            int num_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, low_registers, block_size, 0);

            int warps_per_sm = num_blocks * (block_size / 32);
            int max_warps = prop.maxThreadsPerMultiProcessor / 32;
            float occupancy = 100.0f * warps_per_sm / max_warps;

            // Get register count (from nvcc output)
            int regs = 8;

            printf("%-12s | %-8d | %-8d | %-10d | %-10d | %10.1f%%\n",
                   "low_reg", block_size, regs, num_blocks, warps_per_sm, occupancy);

            // Benchmark
            CUDATimer timer;
            std::vector<float> times;

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; i++)
            {
                low_registers<<<grid_size, block_size>>>(d_data, N);
            }
            cudaDeviceSynchronize();

            // Timed runs
            for (int i = 0; i < NUM_RUNS; i++)
            {
                timer.start_timer();
                low_registers<<<grid_size, block_size>>>(d_data, N);
                float ms = timer.stop_timer();
                times.push_back(ms * 1000.0f); // Convert to microseconds
            }

            auto stats = BenchmarkStats::compute(times);
            results.push_back({"low_reg", block_size, regs, num_blocks,
                               warps_per_sm, occupancy, stats.median});
        }

        // High registers kernel
        {
            int num_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks, high_registers, block_size, 0);

            int warps_per_sm = num_blocks * (block_size / 32);
            int max_warps = prop.maxThreadsPerMultiProcessor / 32;
            float occupancy = 100.0f * warps_per_sm / max_warps;

            // Get register count (from nvcc output)
            int regs = 136;

            printf("%-12s | %-8d | %-8d | %-10d | %-10d | %10.1f%%\n",
                   "high_reg", block_size, regs, num_blocks, warps_per_sm, occupancy);

            // Benchmark
            CUDATimer timer;
            std::vector<float> times;

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; i++)
            {
                high_registers<<<grid_size, block_size>>>(d_data, N);
            }
            cudaDeviceSynchronize();

            // Timed runs
            for (int i = 0; i < NUM_RUNS; i++)
            {
                timer.start_timer();
                high_registers<<<grid_size, block_size>>>(d_data, N);
                float ms = timer.stop_timer();
                times.push_back(ms * 1000.0f);
            }

            auto stats = BenchmarkStats::compute(times);
            results.push_back({"high_reg", block_size, regs, num_blocks,
                               warps_per_sm, occupancy, stats.median});
        }
    }

    printf("\n================================================================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("================================================================================\n");
    printf("%-12s | %-8s | %-10s | %-12s | %-12s | %-10s\n",
           "Kernel", "BlkSize", "Occupancy", "Time (us)", "Throughput", "Slowdown");
    printf("--------------------------------------------------------------------------------\n");

    // Find baseline (low_reg with best performance)
    float baseline_time = 1e9;
    for (const auto &r : results)
    {
        if (strcmp(r.name, "low_reg") == 0 && r.time_us < baseline_time)
        {
            baseline_time = r.time_us;
        }
    }

    for (const auto &r : results)
    {
        float throughput_gb = (N * sizeof(float) * 2) / (r.time_us * 1e-6) / 1e9;
        float slowdown = r.time_us / baseline_time;
        printf("%-12s | %-8d | %9.1f%% | %10.2f | %8.2f GB/s | %9.2fx\n",
               r.name, r.block_size, r.occupancy, r.time_us, throughput_gb, slowdown);
    }

    printf("\n================================================================================\n");
    printf("ANALYSIS: OCCUPANCY vs PERFORMANCE\n");
    printf("================================================================================\n");

    // Group by block size and compare
    printf("\nBlock Size | Low Reg Occ | High Reg Occ | Low Time | High Time | Perf Ratio\n");
    printf("---------------------------------------------------------------------------\n");

    for (int bs : block_sizes)
    {
        Result low_r{}, high_r{};
        for (const auto &r : results)
        {
            if (r.block_size == bs)
            {
                if (strcmp(r.name, "low_reg") == 0)
                    low_r = r;
                else
                    high_r = r;
            }
        }

        float perf_ratio = high_r.time_us / low_r.time_us;

        printf("%-10d | %10.1f%% | %11.1f%% | %8.2f | %9.2f | %9.2fx\n",
               bs, low_r.occupancy, high_r.occupancy,
               low_r.time_us, high_r.time_us, perf_ratio);
    }

    printf("\n================================================================================\n");
    printf("KEY INSIGHTS\n");
    printf("================================================================================\n");
    printf("1. Register pressure (136 regs) reduces occupancy by limiting blocks/SM\n");
    printf("2. SM can hold %d regs: %d/136 = %d threads max (~%d warps)\n",
           prop.regsPerMultiprocessor, prop.regsPerMultiprocessor,
           prop.regsPerMultiprocessor / 136, prop.regsPerMultiprocessor / 136 / 32);
    printf("3. Low register kernel can achieve higher occupancy\n");
    printf("4. Performance impact depends on whether kernel is:\n");
    printf("   - Memory-bound: Low occupancy hurts (can't hide latency)\n");
    printf("   - Compute-bound: May not matter (enough ALU work)\n");
    printf("\n");

    // Determine if memory or compute bound
    float low_best_time = 1e9, high_best_time = 1e9;
    for (const auto &r : results)
    {
        if (strcmp(r.name, "low_reg") == 0)
            low_best_time = std::min(low_best_time, r.time_us);
        else
            high_best_time = std::min(high_best_time, r.time_us);
    }

    float low_bw = (N * sizeof(float) * 2) / (low_best_time * 1e-6) / 1e9;
    float high_bw = (N * sizeof(float) * 2) / (high_best_time * 1e-6) / 1e9;

    printf("Effective bandwidth - Low reg: %.1f GB/s, High reg: %.1f GB/s\n", low_bw, high_bw);
    printf("(Peak memory bandwidth varies by GPU - A100: ~1555 GB/s, V100: ~900 GB/s)\n");

    if (high_best_time > low_best_time * 2.0f)
    {
        printf("\nCONCLUSION: High register pressure significantly hurts performance.\n");
        printf("           The kernel appears to be MEMORY-BOUND - low occupancy\n");
        printf("           prevents hiding memory latency.\n");
    }
    else if (high_best_time > low_best_time * 1.2f)
    {
        printf("\nCONCLUSION: Moderate performance impact from register pressure.\n");
        printf("           Kernel has mixed characteristics.\n");
    }
    else
    {
        printf("\nCONCLUSION: Register pressure has minimal performance impact!\n");
        printf("           The kernel is likely COMPUTE-BOUND - enough ALU work\n");
        printf("           to hide latency even with low occupancy.\n");
    }

    // Sweet spot analysis
    printf("\n================================================================================\n");
    printf("SWEET SPOT ANALYSIS\n");
    printf("================================================================================\n");
    printf("Finding optimal block size for each kernel...\n\n");

    float best_low_time = 1e9, best_high_time = 1e9;
    int best_low_bs = 0, best_high_bs = 0;
    float best_low_occ = 0, best_high_occ = 0;

    for (const auto &r : results)
    {
        if (strcmp(r.name, "low_reg") == 0 && r.time_us < best_low_time)
        {
            best_low_time = r.time_us;
            best_low_bs = r.block_size;
            best_low_occ = r.occupancy;
        }
        if (strcmp(r.name, "high_reg") == 0 && r.time_us < best_high_time)
        {
            best_high_time = r.time_us;
            best_high_bs = r.block_size;
            best_high_occ = r.occupancy;
        }
    }

    printf("Low-register kernel:  Best block size = %d, Occupancy = %.1f%%, Time = %.2f us\n",
           best_low_bs, best_low_occ, best_low_time);
    printf("High-register kernel: Best block size = %d, Occupancy = %.1f%%, Time = %.2f us\n",
           best_high_bs, best_high_occ, best_high_time);
    printf("\nThe 'sweet spot' depends on your kernel's compute/memory ratio:\n");
    printf("- Memory-bound kernels: Maximize occupancy (use fewer registers)\n");
    printf("- Compute-bound kernels: Can trade occupancy for more registers/thread\n");

    cudaFree(d_data);
}

int main()
{
    printf("=== Register Pressure & Occupancy Analysis ===\n\n");
    run_reg_pressure_analysis();
    return 0;
}