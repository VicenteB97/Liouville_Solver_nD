#ifndef __CUDABASE_CUH__
#define __CUDABASE_CUH__

#define deviceFunction __device__
#define hostFunction __host__
#define gpuDevice cudaDevice

// Headers for the CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Thrust libraries
#include <thrust/host_vector.h>         // To use the host vector class. Equivalent to std::vector
#include <thrust/device_vector.h>       // GPU-equivalent of std::vector
#include <thrust/reduce.h>              // To compute the parallel reduction of vectors
#include <thrust/execution_policy.h>    // Important to specify where the reduction is taking place
#include <thrust/functional.h>          // OPTIONAL?
#include <thrust/transform.h>           // To multiply vectors using a parallel reduction-like fashion
#include <thrust/fill.h>                // Explicitly fill host/device functions with a value
#include <thrust/sort.h>                // GPU-based sorting
#include <thrust/extrema.h>             // Compute max. and min. of GPU arrays
#include <thrust/inner_product.h>		// To compute the inner product of GPU arrays

template<typename T>
__global__ static void device_launch_function_wrapper(const T functor) {
	functor(threadIdx.x + blockDim.x + blockIdx.x);
};


class cudaDevice {
public:
	int device_malloc(void** ptr, uint64_t size_bytes) {
		return cudaMalloc(ptr, size_bytes);
	};

	void memCpy_host_to_device(void* dst_dvc, void* src_hst, uint64_t size_bytes) {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_dvc, src_hst, size_bytes, cudaMemcpyHostToDevice);
	};

	void memCpy_device_to_host(void* dst_hst, void* src_dvc, uint64_t size_bytes) {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_hst, src_dvc, size_bytes, cudaMemcpyDeviceToHost);
	};

	void memCpy_device_to_device(void* dst_dvc, void* src_dvc, uint64_t size_bytes) {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_dvc, src_dvc, size_bytes, cudaMemcpyDeviceToDevice);
	};

	void device_free(void* ptr) {
		cudaFree(ptr);
	};

	template<typename T>
	void launch_kernel(uint64_t n_blocks, uint32_t n_threads, const T& functor) const {
		device_launch_function_wrapper <<< n_blocks, n_threads >>> (functor);
	};
};

#endif