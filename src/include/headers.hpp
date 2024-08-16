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
#include <omp.h>        // C library for simple use of thread-parallel computations
#include <chrono>       // Timing the execution of functions
#include <limits>		// To get limits from variables
#include <thread>		// For multithreading
#include <atomic>		// For multithreading atomic functions
#include <stdexcept>	// For handling C++ exceptions
#include <memory>		// For unique, shared and weak pointers

#include "Case_definition.cuh"
#include "utils/numeric_defs.hpp"
#include "cudaBase.cuh"

static const gpuDevice gpu_device;	// We declare the gpu_device at the beggining

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif

/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a numerator
/// @param b denominator
/// @return positive mod(a,b) value
hostFunction deviceFunction
inline uint64_t positive_rem(const uint64_t a, const uint64_t b) {
	return (a % b + b) % b;
}

#endif