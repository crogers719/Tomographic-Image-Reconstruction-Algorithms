# NVCC_FLAGS += -O3 -gencode arch=compute_30,code=sm_30 -lineinfo -std=c++11
NVCC_FLAGS += -O3 -gencode arch=compute_35,code=sm_35 -lineinfo -std=c++11 -rdc=true -lcudadevrt

cuda_files := $(wildcard *cu)
test_files := $(wildcard tests/*cu)
ptr_files := $(shell grep '\#include "cuda_ptr.cuh"' -l *.cu)
io_files := $(shell grep '\#include "mimo-io.cuh"' -l *.cu)
gen_files := $(shell grep '\#include "generator.cuh"' -l *.cu)
rand_files := $(shell grep '\#include <curand_kernel.h>' -l *.cu)

cuda_programs := $(cuda_files:.cu=.exe)
test_programs := $(test_files:.cu=.exe)
ptr_programs := $(ptr_files:.cu=.exe)
io_programs := $(io_files:.cu=.exe)
gen_programs := $(gen_files:.cu=.exe)
debug_programs := $(ptr_files:.cu=-d.exe)

all: $(cuda_programs) $(debug_programs)
tests: $(test_programs)

$(debug_programs): NVCC_FLAGS += -DDEBUG
$(debug_programs) $(ptr_programs): cuda_ptr.cuh
$(io_programs): mimo-io.cuh
$(gen_programs): generator.cuh

%-d.exe: %.cu
	nvcc $(NVCC_FLAGS) $< -o $@

%.exe: %.cu
	nvcc $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.exe tests/*.exe

print-%: 
	@echo $* = $($*)

