#ifndef __CLASSES_CUH__
#define __CLASSES_CUH__


// ------------------------------------------------------------------------------- //
//  HERE ARE THE DEFINITIONS THAT CAN BE CHANGED ACCORDING TO THE PROBLEM TO STUDY //
#define CASE 2				// What case you want to simulate! 1 or 2 (1 is only ready to go with impulses, otherwise you should change the conditions appropriately)
#if (CASE == 1)
	#define DIMENSIONS 1		// SPATIAL DIMENSIONS = NUMBER OF EQUATIONS OF RDE SYSTEM
	#define PARAM_DIMENSIONS 2	// PARAMETER SPACE DIMENSIONS = NUMBER OF PARAMETERS (ICs MUST NOT BE INCLUDED)
	#define IMPULSE true		// WHETHER DELTA-LIKE IMPULSE TERMS ARE TO BE ADDED
#endif
#if(CASE == 2)
	#define DIMENSIONS 2		// SPATIAL DIMENSIONS = NUMBER OF EQUATIONS OF RDE SYSTEM
	#define PARAM_DIMENSIONS 2	// PARAMETER SPACE DIMENSIONS = NUMBER OF PARAMETERS (ICs MUST NOT BE INCLUDED)
	#define IMPULSE true		// WHETHER DELTA-LIKE IMPULSE TERMS ARE TO BE ADDED
#endif
// ------------------------------------------------------------------------------- //
// ------------------------------------------------------------------------------- //

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

#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>

#include <chrono>

#define THREADS_P_BLK 256

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

class Impulse_Param_vec {
public:
	double sample_vec[DIMENSIONS];
	double Joint_PDF;
};

class Param_pair {
public:
	double sample, PDF;
};

class Distributions{
public:
	double params[2];
	char Name;
	bool Truncated;
	double trunc_interval[2];
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
__device__ inline void _D_atomicAdd(double *address, double value) {

    unsigned long long oldval, newval, readback;

	oldval = __double_as_longlong(*address);

    newval = __double_as_longlong(__longlong_as_double(oldval) + value);

    while ((readback = atomicCAS((unsigned long long *) address, oldval, newval)) != oldval) {

        oldval = readback;

        newval = __double_as_longlong(__longlong_as_double(oldval) + value);

    }

}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //


// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
__host__ __device__ double Distance(const gridPoint P1, const gridPoint P2) {

	double out = 0;

	#pragma unroll
	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out += (P1.dim[d] - P2.dim[d]) * (P1.dim[d] - P2.dim[d]);
	}

	return sqrt(out);
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
__host__ __device__ gridPoint Mult_by_Scalar(double scalar, gridPoint Point) {
	gridPoint out;

	#pragma unroll
	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out.dim[d] = scalar * Point.dim[d];
	}

	return out;
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

#endif