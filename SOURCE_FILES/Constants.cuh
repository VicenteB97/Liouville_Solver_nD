#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

// Common C++/C libraries used
#include <fstream>      // Files output
#include <iostream>     // Command Line input/output
#include <vector>       // vector class is widely used
#include <cmath>        // C library (used for mathematical stuff)
#include <numeric>      // OPTIONAL?
#include <algorithm>    // OPTIONAL?
#include <cstdint>      // C library. Allows the use of specific int and unsigned int types, such as uint16_t
#include <string>       // For strings
#include <omp.h>        // For simple use of thread-parallel computations
#include <chrono>       // Timing the execution of functions

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

// Include the Simulation definition!
#include "Case_definition.cuh"			// Here, we read the information of the problem to be simulated!

// Simulation and logging default output folder
#define SIM_OUTPUT_relPATH "../../SIMULATION_OUTPUT/"
#define LOG_OUTPUT_relPATH "../../SIMULATION_OUTPUT/"

// Compute + storage type
#define TYPE float

// Maximum simulation file size
#define MAX_FILE_SIZE_B (uint64_t) 4*1024*1024*1024

// Check the dimensionality of the simulations to use "optimal" data types
#if PHASE_SPACE_DIM < 4
	#define INT int32_t
	#define UINT uint32_t
#else
	#define INT int64_t
	#define UINT uint64_t
#endif

// Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// Mathematical definitions we need
#define M_PI 3.14159265358979323846
#define ptSEARCH_THRESHOLD 50000 // The min. number of particles per sample where we use the counting sort search (not worth it yet)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// GPU-version of the Assert function
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort = true){
if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Some more defines that shouldn't be modified
#define gpuError_Check(ans) {gpuAssert((cudaError_t) ans, __FILE__, __LINE__);}

// This is for the thrust library
#define rpc(ans,offset) raw_pointer_cast(&ans[offset])

// This function is widely used throughout the code so...quite effective
/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a This is the numerator
/// @param b This is the denominator
/// @return Returns mod(a,b)
__host__ __device__
inline UINT positive_rem(const INT a, const INT b) {
	return (a % b + b) % b;
}

/// @brief 
/// @param inputTerminal 
/// @return 
inline bool isNumeric(const std::string& inputTerminal){

	// Iterate through every character in the string and check if they're numbers
	int16_t strLength = inputTerminal.size(), CharPosition = 0;
	bool isDecimal = false;

	// It's OK if the first term is a minus sign
	if(inputTerminal[0] == '-'){
		CharPosition++;
	}
	// It's OK if the first term is a dot character
	else if(inputTerminal[0] == '.'){
		CharPosition++;
		isDecimal = true;
	}

	// Go through all characters in the string
	while(CharPosition < strLength){

		// Check if the character is a digit
		if(!std::isdigit(inputTerminal[CharPosition])){
			
			// Check if it's the dot character and wether it has already appeared
			if(inputTerminal[CharPosition] == '.' && !isDecimal){
				isDecimal = true;
				CharPosition++;
			}
			else{
				return false;
			}
		}

		CharPosition++;
	}
	return true;

}

#endif