/*
 * bandwidth_test.cu -- Measure effective HBM (device memory) bandwidth
 * via a copy kernel. A useful tool for:
 *   - Sanity-checking a new GPU or driver install
 *   - Comparing effective vs theoretical bandwidth
 *   - Spotting thermal throttling (bandwidth drops after sustained use)
 *
 * Build: nvcc -O3 -arch=sm_70 bandwidth_test.cu -o bandwidth_test
 * Run:   ./bandwidth_test
 *
 * On an A100 (1.55 TB/s theoretical) you'd expect ~1.3-1.4 TB/s
 * effective here. On an H100 expect ~2.8-3.0 TB/s. On a T4 ~280 GB/s.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        if (_err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_err));\
            return 1;                                                    \
        }                                                                \
    } while (0)

// A saturating copy kernel: one load + one store per element.
// With a large enough array we're bottlenecked by HBM, not compute.
__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst,
                            size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    // 1 GiB per buffer -> big enough to exceed cache.
    constexpr size_t N = (1ull << 30) / sizeof(float);   // 256M floats
    constexpr size_t BYTES = N * sizeof(float);

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Device: %s\n", props.name);
    printf("Buffer size: %.2f GiB per array\n", BYTES / (double)(1 << 30));

    float *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst, BYTES));
    CUDA_CHECK(cudaMemset(d_src, 0, BYTES));

    constexpr int THREADS = 256;
    int blocks = 2 * props.multiProcessorCount; // oversubscribe SMs

    // Warm up (first launch includes JIT / module-load overhead)
    copy_kernel<<<blocks, THREADS>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time over multiple iterations to average out jitter.
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    constexpr int ITERS = 20;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; ++i) {
        copy_kernel<<<blocks, THREADS>>>(d_src, d_dst, N);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // Each iteration moves 2*BYTES across the memory bus (read + write).
    double total_bytes = 2.0 * BYTES * ITERS;
    double seconds = ms / 1000.0;
    double gb_per_s = total_bytes / seconds / 1e9;

    printf("Average kernel time: %.2f ms\n", ms / ITERS);
    printf("Effective bandwidth: %.1f GB/s\n", gb_per_s);

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
