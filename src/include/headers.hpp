#ifndef __HEADERS_HPP__
#define __HEADERS_HPP__

#include "utils/headersCpp.hpp"
#include "Case_definition.hpp"

/*
#ifdef USECUDA
#include "utils/cudaBase.hpp"
#endif
*/

#include "utils/cudaBase.hpp"

static const gpuDevice gpu_device;	// We declare the static gpu_device at the beggining

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif


// Maximum simulation file size
#define MAX_FILE_SIZE_B (uint64_t) 4*1024*1024*1024

// Check the dimensionality of the simulations to use "optimal" data types
#if PHASE_SPACE_DIM < 4
#define intType int32_t
#define uintType uint32_t
#define ERASE_dPDF false
#define ERASE_auxVectors false
#else
#define intType int64_t
#define uintType uint64_t
#define ERASE_dPDF true
#define ERASE_auxVectors false
#endif

// Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// In low-impact CUDA kernels we process more than one element at a time 
#define ELEMENTS_AT_A_TIME 4

// Mathematical definitions we need
#define M_PI 3.14159265358979323846
#define ptSEARCH_THRESHOLD 10000000 // The min. number of particles per sample where we use the counting sort search (not worth it yet)

// Error checking in the CPU code
#define errorCheck(ans) {if(ans != EXIT_SUCCESS){std::cout << "Error found at:\n" << __FILE__ << "\nLine: " << std::to_string(__LINE__) << ".\n"; return EXIT_FAILURE;}}

/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a numerator
/// @param b denominator
/// @return positive mod(a,b) value
hostFunction deviceFunction
inline uint64_t positive_rem(const uint64_t a, const uint64_t b) {
	return (a % b + b) % b;
}

#endif