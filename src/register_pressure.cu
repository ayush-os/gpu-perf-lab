#include <stdio.h>
#include "../include/timer.cuh"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

// Low register pressure (simple kernel)
__global__ void low_registers(float *data, int N)
{
    // Uses ~10 registers
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = data[idx];
        x = x * 2.0f + 1.0f;
        data[idx] = x;
    }
}

__global__ void high_registers(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float x = data[idx];

        // ============================================================
        // PHASE 1: Create 128 variables with sequential dependencies
        // This prevents reordering but compiler might still serialize
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
        // PHASE 3: asm volatile barrier - ALL 128 vars must be live HERE
        // ============================================================
        // asm volatile("" : "+f"(v0), "+f"(v1), "+f"(v2), "+f"(v3), "+f"(v4), "+f"(v5), "+f"(v6), "+f"(v7));
        // asm volatile("" : "+f"(v8), "+f"(v9), "+f"(v10), "+f"(v11), "+f"(v12), "+f"(v13), "+f"(v14), "+f"(v15));
        // asm volatile("" : "+f"(v16), "+f"(v17), "+f"(v18), "+f"(v19), "+f"(v20), "+f"(v21), "+f"(v22), "+f"(v23));
        // asm volatile("" : "+f"(v24), "+f"(v25), "+f"(v26), "+f"(v27), "+f"(v28), "+f"(v29), "+f"(v30), "+f"(v31));
        // asm volatile("" : "+f"(v32), "+f"(v33), "+f"(v34), "+f"(v35), "+f"(v36), "+f"(v37), "+f"(v38), "+f"(v39));
        // asm volatile("" : "+f"(v40), "+f"(v41), "+f"(v42), "+f"(v43), "+f"(v44), "+f"(v45), "+f"(v46), "+f"(v47));
        // asm volatile("" : "+f"(v48), "+f"(v49), "+f"(v50), "+f"(v51), "+f"(v52), "+f"(v53), "+f"(v54), "+f"(v55));
        // asm volatile("" : "+f"(v56), "+f"(v57), "+f"(v58), "+f"(v59), "+f"(v60), "+f"(v61), "+f"(v62), "+f"(v63));
        // asm volatile("" : "+f"(v64), "+f"(v65), "+f"(v66), "+f"(v67), "+f"(v68), "+f"(v69), "+f"(v70), "+f"(v71));
        // asm volatile("" : "+f"(v72), "+f"(v73), "+f"(v74), "+f"(v75), "+f"(v76), "+f"(v77), "+f"(v78), "+f"(v79));
        // asm volatile("" : "+f"(v80), "+f"(v81), "+f"(v82), "+f"(v83), "+f"(v84), "+f"(v85), "+f"(v86), "+f"(v87));
        // asm volatile("" : "+f"(v88), "+f"(v89), "+f"(v90), "+f"(v91), "+f"(v92), "+f"(v93), "+f"(v94), "+f"(v95));
        // asm volatile("" : "+f"(v96), "+f"(v97), "+f"(v98), "+f"(v99), "+f"(v100), "+f"(v101), "+f"(v102), "+f"(v103));
        // asm volatile("" : "+f"(v104), "+f"(v105), "+f"(v106), "+f"(v107), "+f"(v108), "+f"(v109), "+f"(v110), "+f"(v111));
        // asm volatile("" : "+f"(v112), "+f"(v113), "+f"(v114), "+f"(v115), "+f"(v116), "+f"(v117), "+f"(v118), "+f"(v119));
        // asm volatile("" : "+f"(v120), "+f"(v121), "+f"(v122), "+f"(v123), "+f"(v124), "+f"(v125), "+f"(v126), "+f"(v127));

        // ============================================================
        // PHASE 4: Use ALL values (prevents dead code elimination)
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

// Helper to compute and print stats
void print_results(const char *label, float median_us, float baseline_ms = 0.0f)
{
    float slowdown = (baseline_ms > 0) ? (median_us / (baseline_ms * 1000.0f)) : 1.0f;

    printf("%-22s | Time: %8.2f us | Slowdown: %5.2fx\n",
           label, median_us, slowdown);
}

void run_reg_pressure_analysis()
{
    size_t N = 64 * 1024 * 1024;

    float *d_output;
    cudaMalloc(&d_output, N * sizeof(float));

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDATimer timer;
    std::vector<float> times;

    // --- 1. low_regs ---
    for (int i = 0; i < 10; i++)
        low_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        low_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto low_registers_stats = BenchmarkStats::compute(times);
    float baseline_ms = low_registers_stats.median;
    print_results("low_registers", baseline_ms * 1000.0f, baseline_ms);
    times.clear();

    // --- 2. high_registers ---
    for (int i = 0; i < 10; i++)
        high_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < 50; i++)
    {
        timer.start_timer();
        high_registers<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output, N);
        times.push_back(timer.stop_timer());
    }
    auto high_registers_stats = BenchmarkStats::compute(times);
    print_results("high_registers", high_registers_stats.median * 1000.0f, baseline_ms);
    times.clear();

    cudaFree(d_output);
}

int main()
{
    printf("=== Register Pressure Analysis ===\n");
    run_reg_pressure_analysis();
    return 0;
}