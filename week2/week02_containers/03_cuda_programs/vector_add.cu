/*
 * vector_add.cu -- The "hello world" of CUDA programming.
 *
 * Demonstrates:
 *   - Host (CPU) vs device (GPU) memory allocation
 *   - Explicit host-to-device and device-to-host memory copies
 *   - Kernel launch with a grid of blocks of threads
 *   - Basic error checking
 *
 * Build (inside a CUDA-enabled container / dev box):
 *   nvcc -O3 -arch=sm_70 vector_add.cu -o vector_add
 *
 * Run:
 *   ./vector_add
 *
 * Expected output (approximate):
 *   Running on: NVIDIA <your-GPU-name>
 *   Computing c = a + b for N = 1048576 elements
 *   Kernel time: 0.xyz ms
 *   Verification PASSED
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Tiny macro for compact error checking. A real codebase would use
// a library like Thrust or cuCheck.
#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _err = (expr);                                       \
        if (_err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));       \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

// __global__ marks this function as a CUDA kernel: runs on the device,
// callable from the host. Each thread handles one element.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Global thread index: which element does *this* thread handle?
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr int N = 1 << 20;                 // 1,048,576 elements
    constexpr size_t BYTES = N * sizeof(float);

    // Query and print device info.
    int device_id = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    printf("Running on: %s (SM %d.%d)\n",
           props.name, props.major, props.minor);
    printf("Computing c = a + b for N = %d elements\n", N);

    // 1. Allocate host buffers and fill with data.
    float* h_a = (float*) std::malloc(BYTES);
    float* h_b = (float*) std::malloc(BYTES);
    float* h_c = (float*) std::malloc(BYTES);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 2.0f * static_cast<float>(i);
    }

    // 2. Allocate device buffers.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, BYTES));
    CUDA_CHECK(cudaMalloc(&d_b, BYTES));
    CUDA_CHECK(cudaMalloc(&d_c, BYTES));

    // 3. Copy input data host -> device.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice));

    // 4. Configure grid: 256 threads per block is a common sweet spot.
    constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // 5. Time the kernel using CUDA events (nanosecond-ish precision).
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("Kernel time: %.3f ms\n", ms);

    // 6. Copy results back, device -> host.
    CUDA_CHECK(cudaMemcpy(h_c, d_c, BYTES, cudaMemcpyDeviceToHost));

    // 7. Sanity check a few elements.
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            fprintf(stderr, "Mismatch at %d: got %f want %f\n",
                    i, h_c[i], expected);
            ok = false;
            break;
        }
    }
    printf("Verification %s\n", ok ? "PASSED" : "FAILED");

    // 8. Clean up.
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    std::free(h_a); std::free(h_b); std::free(h_c);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ok ? 0 : 1;
}
