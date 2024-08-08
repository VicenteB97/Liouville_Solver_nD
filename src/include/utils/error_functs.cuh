#pragma once

#include "../headers.cuh"

// GPU-version of the Assert function
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort = true){
if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Some more defines that shouldn't be modified
#define gpuError_Check(ans) {gpuAssert((cudaError_t) ans, __FILE__, __LINE__);}

// Error checking in the CPU code
#define errorCheck(ans) {if(ans != 0){std::cout << "Error found at:\n" << __FILE__ << "\nLine: " << std::to_string(__LINE__) << ".\n"; return EXIT_FAILURE;}}
