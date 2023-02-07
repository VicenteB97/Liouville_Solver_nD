#ifndef __CLASSES_CUH__
#define __CLASSES_CUH__
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

//----------------- CLASSES USED ---------------------------------
//----------------------------------------------------------------

// Grid points----------------------------------------------------
class gridPoint { // maybe this should be re-thought
public:
	double dim[DIMENSIONS];

	__host__ __device__ gridPoint operator+(const gridPoint& other) {

		gridPoint out;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			double aux = dim[d];
			aux += other.dim[d];
			out.dim[d] = aux;
		}

		return out;

	}
	__host__ __device__ gridPoint operator-(const gridPoint& other) {
		gridPoint out;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			double aux = dim[d];
			aux -= other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ bool operator==(const gridPoint& other) {
		bool out = true;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			if (dim[d] != other.dim[d]) { out = false; }
		}

		return out;
	}
};

// Time + impulse: ----------------------------------------------
class Time_Impulse_vec {
public:
	double time;
	bool impulse;
	gridPoint impulse_vec;
	double impulse_std[DIMENSIONS];
	int samples[DIMENSIONS];

	bool operator < (const Time_Impulse_vec& other) const {
		return (time < other.time);
	}
};

// Parameters
class Param_vec {
public:
	double sample_vec[PARAM_DIMENSIONS];
	double Joint_PDF;
};

class Param_pair {
public:
	double sample, PDF;
};

//-------------------------------------------------------------------------//
//-------------------------------------------------------------------------//
//-------------------------------------------------------------------------//
//---------------- FUNCTIONS TO BE USED ELSEWHERE IN THE CODE -------------//
//-------------------------------------------------------------------------//
//-------------------------------------------------------------------------//
//-------------------------------------------------------------------------//

// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a This is the numerator
/// @param b This is the denominator
/// @return Returns mod(a,b)
__host__ __device__ inline unsigned int positive_rem(const int a, const int b){
	return (a % b + b) % b;
} 
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
/// @brief This function allows to atomically add values of type double
/// @param address 
/// @param val 
/// @return 
__device__ double atomicAdd_D(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //


// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
__host__ __device__ double Distance(const gridPoint P1, const gridPoint P2) {

	double out = 0;

	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out += (P1.dim[d] - P2.dim[d]) * (P1.dim[d] - P2.dim[d]);
	}

	return sqrt(out);
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
__device__ gridPoint Mult_by_Scalar(double scalar, gridPoint Point) {
	gridPoint out;

	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out.dim[d] = scalar * Point.dim[d];
	}

	return out;
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

#endif