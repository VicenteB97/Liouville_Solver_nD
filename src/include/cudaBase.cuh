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
	uint16_t memCpy_host_to_device(void* dst_dvc, void* src_hst, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_dvc, src_hst, size_bytes, cudaMemcpyHostToDevice);
	};

	uint16_t memCpy_device_to_host(void* dst_hst, void* src_dvc, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		cudaMemcpy(dst_hst, src_dvc, size_bytes, cudaMemcpyDeviceToHost);
	};

	uint16_t memCpy_device_to_device(void* dst_dvc, void* src_dvc, uint64_t size_bytes) const {
		cudaDeviceSynchronize();
		if (cudaSuccess != cudaMemcpy(dst_dvc, src_dvc, size_bytes, cudaMemcpyDeviceToDevice)) {
			std::cout << "Error copying memory between device arrays.\n";
			return EXIT_FAILURE;
		};
	};

	template<typename T>
	void launch_kernel(uint64_t n_blocks, uintType n_threads, const T& functor) const {
		device_launch_function_wrapper <<< n_blocks, n_threads >>> (functor);
		cudaDeviceSynchronize();
	};
};


// Rethink the error output for this class' methods, especially regarding memory allocation and memory value initialization
// Read a little bit about exception vs return codes for performance!! And how/when to launch exceptions

template<typename T>
class cuda_unique_ptr {
private:
	T* __raw_dvc_pointer;
	uint32_t __size_count;

public:
	// Default constructor
	cuda_unique_ptr() : __raw_dvc_pointer(nullptr), __size_count(0)) {};

	// Allocate and initialize constructor
	cuda_unique_ptr(uint32_t size, T init_value = (T)0) : __raw_dvc_pointer(nullptr), __size_count(0) {
		try {
			this->malloc(size);
			this->set_init_values(init_value);
		}
		catch (const std::exception& e) {
			std::cerr << "Error in cuda unique pointer initialization: " << e.what();
			throw;
		}

		// Set size count only if allocation and initialization succeeded
		__size_count = size;
	};

	~cuda_unique_ptr() {	
		try {
			this->__free();
		}
		catch (const std::exception& e) {
			std::cerr << "Error destroying cuda_unique_poitner object: " << e.what() << std::endl;
			// Note that we don't throw errors here! Cannot allow exceptions to exit destructors
		}
	};

	// Delete copy constructors and assignment operators for this class (equivalent to the uniquePtr from std::unique_ptr)
	cuda_unique_ptr(const cuda_unique_ptr&) = delete;	// Copy constructor
	cuda_unique_ptr& operator=(const cuda_unique_ptr&) = delete;	// Assignment operator

	hostFunction
	/// @brief This function allocates size_count elements of memory type given by the template parameter T of the unique_poitner.
	/// This function returns a runtime exception if allocation fails.
	/// @param size_count - The number of elements that the poitner will point to.
	void malloc(uint64_t size_count = 1) {
		if (size <= 0) { throw std::runtime_error("Cannot allocate 0 elements on array.\n") };
		
		// Allocate memory assuming the desired size is at least 1
		if (cudaMalloc((void**)&__raw_dvc_pointer, sizeof(T) * size_count) != cudaSuccess) {
			throw std::runtime_error("Error allocating device memory.\n");
		};
	};

	hostFunction
	/// @brief This function sets the initial values of the array the cuda pointer points to. Throws an exception if memory value setting goes wrong
	/// @param value - Initial value set to the array
	/// @param first_element - First array element pointed
	/// @param last_element - Last array element poitned to
	void set_init_values(T value = 0, uint64_t first_element = 0, uint64_t last_element = __size_count) {
		if (value == (T)0) {
			if (cudaMemset((void*)__raw_dvc_pointer, (int32_t)0, __size_count * sizeof(T)) != cudaSuccess) {
				throw std::runtime_error("Failed to initialize device memory with cudaMemset");
			}
		}
		else {
			try {
				thrust::device_ptr<T> aux_dev_ptr(__raw_dvc_pointer);
				thrust::fill(aux_dev_ptr + first_element, aux_dev_ptr + last_element, value);
			}
			catch (const std::exception& e) {
				std::cerr << "Cannot set initial value in device variable using thrust::fill.\n";
				throw;
			}
		}
	};

	/// @brief Returns the raw device pointer
	hostFunction
		T* get() const {
		return __raw_dvc_pointer;
	};

	/// @brief Get the size (number of elements) of the pointed array
	hostFunction
	uint64_t size_count() const {
		return __size_count;
	};

	/// @brief Get the size (in bytes) of the pointed array
	hostFunction
	uint64_t size_bytes() const {
		return __size_count * sizeof(T);
	};

private:
	hostFunction
	void __free() {
		__size_count = 0;
		if (cudaFree(__raw_dvc_pointer) != cudaSuccess){ // cudaFree automatically manages the case of the nullptr, so no need to check here
			throw std::exception("Error when deallocating device memory.\n");
		}
		__raw_dvc_pointer = nullptr;
	};
};

#endif