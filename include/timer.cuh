#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

class CUDATimer
{
private:
    cudaEvent_t start, stop;

public:
    CUDATimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start_timer()
    {
        cudaEventRecord(start);
    }

    float stop_timer()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Statistical analysis helper
struct BenchmarkStats
{
    float mean;
    float stddev;
    float min;
    float max;
    float median;

    static BenchmarkStats compute(std::vector<float> &times)
    {
        BenchmarkStats ret;

        if (times.empty())
            return ret;

        // Mean
        float sum = std::accumulate(times.begin(), times.end(), 0.0f);
        float mean = sum / times.size();

        // Min and Max
        auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());
        float min_val = *min_it;
        float max_val = *max_it;

        // standard deviation
        float variance = 0.0f;
        for (float x : times)
        {
            variance += std::pow(x - mean, 2);
        }
        variance /= times.size();
        float stddev = std::sqrt(variance);

        // median
        size_t n = times.size() / 2;
        std::nth_element(times.begin(), times.begin() + n, times.end());
        float median;

        if (times.size() % 2 != 0)
        {
            // Odd number of elements
            median = times[n];
        }
        else
        {
            // Even number: average of the two middle elements
            float val1 = times[n];
            auto it = std::max_element(times.begin(), times.begin() + n);
            float val2 = *it;
            median = (val1 + val2) / 2.0f;
        }

        ret.mean = mean;
        ret.stddev = stddev;
        ret.min = min_val;
        ret.max = max_val;
        ret.median = median;
    }

    void print()
    {
        printf("Mean: %.3f μs, Stddev: %.3f μs, Min: %.3f μs, Max: %.3f μs, Median: %.3f μs\n",
               mean, stddev, min, max, median);
    }
};

#endif