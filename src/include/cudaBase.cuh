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
	int device_malloc(void** ptr, uint64_t size_bytes) const {
		return cudaMalloc(ptr, size_bytes);
	};

	void memCpy_host_to_device(void* dst_dvc, void* src_hst, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_dvc, src_hst, size_bytes, cudaMemcpyHostToDevice);
	};

	void memCpy_device_to_host(void* dst_hst, void* src_dvc, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_hst, src_dvc, size_bytes, cudaMemcpyDeviceToHost);
	};

	void memCpy_device_to_device(void* dst_dvc, void* src_dvc, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_dvc, src_dvc, size_bytes, cudaMemcpyDeviceToDevice);
	};

	void device_memSet(void* ptr, intType value, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		cudaMemset(ptr, value, size_bytes);
	};

	template<typename T>
	void launch_kernel(uint64_t n_blocks, uintType n_threads, const T& functor) const {
		device_launch_function_wrapper <<< n_blocks, n_threads >>> (functor);
		cudaDeviceSynchronize();
	};
};


template<typename T>
class cuda_unique_ptr {
private:
	T* __raw_dvc_pointer;
	uint32_t __size_count;
	T __init_value;

public:
	cuda_unique_ptr() : __raw_dvc_pointer(nullptr), __size_count(0), __init_value((T)0) {};

	//cuda_unique_ptr(uint32_t size, T init_value) :
	//	__raw_dvc_pointer(nullptr), __size_count(size), __init_value(init_value) {
	//	this->allocate_and_init();
	//};

	~cuda_unique_ptr() {
		this->free();
	};

	// Delete copy constructors and assignment operators for this class (I want to "implement" the uniquePtr from std)
	cuda_unique_ptr(const cuda_unique_ptr&) = delete;
	cuda_unique_ptr& operator=(const cuda_unique_ptr&) = delete;

	/// @brief Returns the raw device pointer
	hostFunction
	T* get() const {
		return __raw_dvc_pointer;
	};

	hostFunction
	uint64_t size_count() const {
		return __size_count;
	};

	hostFunction
	uint64_t size_bytes() const {
		return __size_count * sizeof(T);
	};

private:
	hostFunction
	void free() noexcept {
		if (__raw_dvc_pointer != nullptr) {
			cudaFree(__raw_dvc_pointer);
			__raw_dvc_pointer = nullptr;
		}
	};
};

#endif