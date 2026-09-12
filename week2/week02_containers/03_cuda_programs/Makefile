NVCC     := nvcc
# -arch=native picks the SM version of the GPU in the build container.
# For explicit Blackwell target: -gencode arch=compute_100,code=sm_100
NVCCFLAGS := -arch=native -O2
TARGETS  := vector_add bandwidth_test

.PHONY: all clean

all: $(TARGETS)

vector_add: vector_add.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

bandwidth_test: bandwidth_test.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
