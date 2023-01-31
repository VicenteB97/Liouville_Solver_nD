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
	double position[DIMENSIONS];

	__host__ __device__ gridPoint operator+(const gridPoint& other) {

		gridPoint out;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			double aux = position[d];
			aux += other.position[d];
			out.position[d] = aux;
		}

		return out;

	}
	__host__ __device__ gridPoint operator-(const gridPoint& other) {
		gridPoint out;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			double aux = position[d];
			aux -= other.position[d];
			out.position[d] = aux;
		}

		return out;
	}
	__host__ __device__ bool operator==(const gridPoint& other) {
		bool out = true;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			if (position[d] != other.position[d]) { out = false; }
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

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//------------------ FUNCTIONS TO BE USED ELSEWHERE IN THE CODE------------
//-------------------------------------------------------------------------

// Computing the mod of two numbers (AVOIDING NEGATIVE REMAINDERS)
inline unsigned int positive_rem(const int a, const int b){
	return (a % b + b) % b;
} 

/// <summary>
/// Computes euclidean distance between two points in space
/// </summary>
/// <param name="P1"></param>
/// <param name="P2"></param>
/// <returns></returns>
__host__ __device__ double Distance(const gridPoint P1, const gridPoint P2) {

	double out = 0;

	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out += (P1.position[d] - P2.position[d]) * (P1.position[d] - P2.position[d]);
	}

	return sqrt(out);
}

/// <summary>
/// Computes the multiplication of a scalar with a point in space
/// </summary>
/// <param name="scalar"></param>
/// <param name="Point"></param>
/// <returns></returns>
__device__ gridPoint Mult_by_Scalar(double scalar, gridPoint Point) {
	gridPoint out;

	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out.position[d] = scalar * Point.position[d];
	}

	return out;
}

#endif