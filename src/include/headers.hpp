#ifndef __HEADERS_HPP__
#define __HEADERS_HPP__

// C++/C libraries used
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
#include <limits>		// To get limits from variables
#include <thread>		// For multithreading
#include <atomic>		// For multithreading atomic functions

#include "Case_definition.cuh"
#include "cudaBase.cuh"

static constexpr uint16_t THREADS_P_BLK = 128;

static const gpuDevice gpu_device;

// This is for the thrust library
#define rpc(ans,offset) raw_pointer_cast(&ans[offset])

// This function is widely used throughout the code so...quite effective
/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a This is the numerator
/// @param b This is the denominator
/// @return Returns mod(a,b)
hostFunction deviceFunction
inline uint64_t positive_rem(const uint64_t a, const uint64_t b) {
	return (a % b + b) % b;
}

#endif