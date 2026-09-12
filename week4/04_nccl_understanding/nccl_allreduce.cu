/*
 * nccl_allreduce.cu -- Illustrative raw NCCL all-reduce from C/CUDA,
 * without MPI or PyTorch. Shows what PyTorch DDP is doing under the
 * hood: create a communicator, post collectives on a stream, sync.
 *
 * Single-process, multi-GPU variant: one rank per GPU on one machine.
 * In real code you'd use one process per GPU and ncclGetUniqueId +
 * MPI_Bcast or a TCP rendezvous to share the id across processes.
 *
 * Build (inside an NVIDIA container with NCCL headers):
 *   nvcc -O3 -arch=sm_80 nccl_allreduce.cu -lnccl -o nccl_allreduce
 *
 * Run (e.g. on an 8-GPU box):
 *   ./nccl_allreduce 8 1048576      # 8 GPUs, 1M floats
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA(expr)                                                      \
    do { cudaError_t e = (expr);                                        \
         if (e != cudaSuccess) {                                        \
             fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e));       \
             std::exit(1);                                              \
         } } while (0)

#define NCCL(expr)                                                      \
    do { ncclResult_t r = (expr);                                       \
         if (r != ncclSuccess) {                                        \
             fprintf(stderr, "NCCL %s\n", ncclGetErrorString(r));       \
             std::exit(1);                                              \
         } } while (0)

int main(int argc, char** argv) {
    int nranks  = (argc > 1) ? atoi(argv[1]) : 2;
    int n_elems = (argc > 2) ? atoi(argv[2]) : (1 << 20);

    // Per-rank CUDA streams and device pointers.
    std::vector<cudaStream_t> streams(nranks);
    std::vector<float*>       buffers(nranks);
    std::vector<ncclComm_t>   comms(nranks);

    // Allocate a buffer on each GPU and seed with the rank index so
    // the all-reduce sum is 0+1+...+(N-1) = N*(N-1)/2 per element.
    for (int r = 0; r < nranks; ++r) {
        CUDA(cudaSetDevice(r));
        CUDA(cudaMalloc(&buffers[r], n_elems * sizeof(float)));
        CUDA(cudaStreamCreate(&streams[r]));
        std::vector<float> host(n_elems, static_cast<float>(r));
        CUDA(cudaMemcpy(buffers[r], host.data(),
                        n_elems * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialise communicators. For single-process multi-GPU the
    // convenience API ncclCommInitAll handles unique-id generation.
    NCCL(ncclCommInitAll(comms.data(), nranks, /*devlist*/ nullptr));

    // Issue the collective. NCCL groups the per-rank calls into one
    // operation between ncclGroupStart/End.
    NCCL(ncclGroupStart());
    for (int r = 0; r < nranks; ++r) {
        NCCL(ncclAllReduce(buffers[r], buffers[r],
                           n_elems, ncclFloat, ncclSum,
                           comms[r], streams[r]));
    }
    NCCL(ncclGroupEnd());

    // Wait for each stream; then verify the result on device 0.
    for (int r = 0; r < nranks; ++r) {
        CUDA(cudaSetDevice(r));
        CUDA(cudaStreamSynchronize(streams[r]));
    }

    std::vector<float> out(n_elems);
    CUDA(cudaSetDevice(0));
    CUDA(cudaMemcpy(out.data(), buffers[0],
                    n_elems * sizeof(float), cudaMemcpyDeviceToHost));
    float expected = nranks * (nranks - 1) / 2.0f;
    bool ok = true;
    for (int i = 0; i < 4; ++i) {     // spot-check
        if (out[i] != expected) { ok = false; break; }
    }
    printf("all-reduce of %d GPUs * %d floats : %s (first=%f expected=%f)\n",
           nranks, n_elems, ok ? "OK" : "FAIL", out[0], expected);

    // Cleanup.
    for (int r = 0; r < nranks; ++r) {
        ncclCommDestroy(comms[r]);
        cudaFree(buffers[r]);
        cudaStreamDestroy(streams[r]);
    }
    return ok ? 0 : 1;
}
