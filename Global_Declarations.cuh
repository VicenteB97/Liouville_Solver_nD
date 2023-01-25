#ifndef __GLOBAL_DECLARATIONS_CUH__
#define __GLOBAL_DECLARATIONS_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/fill.h>

#include <chrono>

#define DIMENSIONS 2		
#define PARAM_DIMENSIONS 2
#define THREADS_P_BLK 512

// This is for CUDA built-in functions error handling
#define gpuError_Check(ans) {gpuAssert((cudaError_t) ans, __FILE__, __LINE__);}

inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort = true){
if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif