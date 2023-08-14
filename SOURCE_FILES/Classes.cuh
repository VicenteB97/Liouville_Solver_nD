//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
// This part contains the definitions of the classes and some functions that will be used elsewhere in our code//
// We STRONGLY RECOMMEND not changing absolutely anything if your only use of this program is to simulate	   //
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
#ifndef __CLASSES_CUH__
#define __CLASSES_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <string>
#include <omp.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include <chrono>

#include "Case_definition.cuh"

// Some more defines that shouldn't be modified
#define gpuError_Check(ans) {gpuAssert((cudaError_t) ans, __FILE__, __LINE__);}
#define rpc(ans,offset) raw_pointer_cast(&ans[offset])

#define RELATIVE_PATH "../../SIMULATION_OUTPUT/"

#define FIXED_TYPE double

inline void gpuAssert (cudaError_t code, const char *file, int32_t line, bool abort = true){
if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

# define M_PI 3.14159265358979323846

//-------------------------------------------------------------------------//
//-------------------------- CLASSES USED ---------------------------------//
//-------------------------------------------------------------------------//
// Grid points----------------------------------------------------
class gridPoint {
public:
	TYPE dim[DIMENSIONS];

	__host__ __device__ gridPoint operator+(const gridPoint& other) {

		gridPoint out;

		for (uint32_t d = 0; d < DIMENSIONS; d++) {
			TYPE aux = dim[d];
			aux += other.dim[d];
			out.dim[d] = aux;
		}

		return out;

	}
	__host__ __device__ gridPoint operator-(const gridPoint& other) {
		gridPoint out;

		for (uint32_t d = 0; d < DIMENSIONS; d++) {
			TYPE aux = dim[d];
			aux -= other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ bool operator==(const gridPoint& other) {
		bool out = true;

		for (uint32_t d = 0; d < DIMENSIONS; d++) {
			if (dim[d] != other.dim[d]) { out = false; }
		}

		return out;
	}
};

// Time + impulse: ----------------------------------------------
class Time_Impulse_vec {
public:
	FIXED_TYPE 		time;
	bool 		impulse;

	bool operator < (const Time_Impulse_vec& other) const {
		return (time < other.time);
	}
};

class AMR_node_select{
public:
	uint32_t node, AMR_selected;

	__host__ __device__ bool operator < (const AMR_node_select& other) const { // Note that we have changed order, for simpler work
		return (AMR_selected > other.AMR_selected);
	}
};

// Parameters
class Param_vec {
public:
	TYPE sample_vec[PARAM_DIMENSIONS];
	TYPE Joint_PDF;
};

class Impulse_Param_vec {
public:
	TYPE sample_vec[DIMENSIONS];
	TYPE Joint_PDF;
};

class Param_pair {
public:
	TYPE sample, PDF;
};

class Distributions{
public:
	TYPE params[2];
	char  Name;
	bool  Truncated;
	TYPE trunc_interval[2];
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
__host__ __device__ inline uint32_t positive_rem(const int32_t a, const int32_t b){
	return (a % b + b) % b;
} 
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //


// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
__host__ __device__ inline TYPE Distance(const gridPoint P1, const gridPoint P2) {

	TYPE out = 0;

	#pragma unroll
	for (uint32_t d = 0; d < DIMENSIONS; d++) {
		out += (P1.dim[d] - P2.dim[d]) * (P1.dim[d] - P2.dim[d]);
	}

	return sqrtf(out);
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //
__host__ __device__ inline gridPoint Mult_by_Scalar(TYPE scalar, gridPoint Point) {
	gridPoint out;

	#pragma unroll
	for (uint32_t d = 0; d < DIMENSIONS; d++) {
		out.dim[d] = scalar * Point.dim[d];
	}

	return out;
}
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

__host__ __device__ inline Param_vec _Gather_Param_Vec(const uint32_t index, const Param_pair* Parameter_Array, const int32_t* n_Samples){

	Param_vec Output;

	Output.Joint_PDF = 1;

	uint32_t aux_samples_mult = 1;
	uint32_t aux_samples_sum  = 0;

	#pragma unroll
	for (uint32_t d = 0; d < PARAM_DIMENSIONS; d++){
		uint32_t aux3 = n_Samples[d];
		uint32_t aux = floorf(positive_rem(index, aux3 * aux_samples_mult) / aux_samples_mult );

		Output.sample_vec[d] = Parameter_Array[aux + aux_samples_sum].sample;
		Output.Joint_PDF 	*= Parameter_Array[aux + aux_samples_sum].PDF;

		aux_samples_mult *= aux3;
		aux_samples_sum  += aux3;
	}
	return Output;
}

__device__ inline bool __is_in_domain(const gridPoint particle, const gridPoint* Boundary){
	bool out = true;
	uint16_t d = 0;

	while(out && d < DIMENSIONS){
		if(particle.dim[d] < Boundary[0].dim[d] || particle.dim[d] > Boundary[1].dim[d]){out = false;}
		d++;
	}

	return out;
}

__device__ __forceinline__ void __atomicAdd(double *address, double val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    unsigned long long *ptr = (unsigned long long *)address;
    unsigned long long old, newdbl, ret = *ptr;
    do {
        old = ret;
        newdbl = __double_as_longlong(__longlong_as_double(old)+val);
    } while((ret = atomicCAS(ptr, old, newdbl)) != old);
}

__device__ __forceinline__ void __atomicAdd(float *address, float val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    unsigned int *ptr = (unsigned int *)address;
    unsigned int old, newint, ret = *ptr;
    do {
        old = ret;
        newint = __float_as_int(__int_as_float(old)+val);
    } while((ret = atomicCAS(ptr, old, newint)) != old);
}

// +===========================================================================+ //
// +===========================================================================+ //
// +===========================================================================+ //
// +===========================================================================+ //

__device__ inline gridPoint VECTOR_FIELD(gridPoint X, TYPE t, const Param_vec parameter, const uint32_t mode, const FIXED_TYPE extra_param[]) {
	gridPoint output = VEC_FIELD;

	return output;
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__ inline TYPE DIVERGENCE_FIELD(gridPoint X, TYPE t, const Param_vec parameter, const uint32_t mode, const FIXED_TYPE extra_param[]) {
	TYPE output = DIVERGENCE;

	return output;
}


#endif