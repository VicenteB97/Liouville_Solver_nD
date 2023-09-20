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
#include <iostream>
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
#include <thrust/extrema.h>

#include <chrono>

#include "Case_definition.cuh"

// Some more defines that shouldn't be modified
#define gpuError_Check(ans) {gpuAssert((cudaError_t) ans, __FILE__, __LINE__);}
#define rpc(ans,offset) raw_pointer_cast(&ans[offset])

#define RELATIVE_PATH "../../SIMULATION_OUTPUT/"

#define FIXED_TYPE double

#define MAX_FILE_SIZE_B (uint64_t)2*1024*1024*1024

#if DIMENSIONS < 3
	#define INT int32_t
	#define UINT uint32_t
#elif DIMENSIONS < 6
	#define INT int64_t
	#define UINT uint64_t
#else
	#define INT int128_t
	#define UINT uint128_t
#endif

inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort = true){
if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define M_PI 3.14159265358979323846
#define ptSEARCH_THRESHOLD 15000 // The min. number of particles per sample where we use the counting sort search

//-------------------------------------------------------------------------//
//-------------------------- CLASSES USED ---------------------------------//
//-------------------------------------------------------------------------//
// Grid points----------------------------------------------------
class gridPoint {
public:
	TYPE dim[DIMENSIONS];

	__host__ __device__ gridPoint operator+(const gridPoint& other) {

		gridPoint out;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			TYPE aux = dim[d];
			aux += other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ gridPoint operator-(const gridPoint& other) {
		gridPoint out;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			TYPE aux = dim[d];
			aux -= other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ bool operator==(const gridPoint& other) {
		bool out = true;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			if (dim[d] != other.dim[d]) { out = false; }
		}

		return out;
	}

	__host__ __device__ inline TYPE Distance(const gridPoint& other) {
		TYPE dist = 0;
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			dist += (dim[d] - other.dim[d]) * (dim[d] - other.dim[d]);
		}
		return sqrtf(dist);
	};

	
	__host__ __device__ 
	inline gridPoint Mult_by_Scalar(TYPE scalar) {
		gridPoint out;

		
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			out.dim[d] = scalar * dim[d];
		}

		return out;
	};

	__device__ inline bool is_in_domain(const gridPoint* Boundary) {
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			if (dim[d] < Boundary[0].dim[d] || dim[d] > Boundary[1].dim[d]) { return false; }
		}
		return true;
	};

	// This function tells us what is the first bin to search if we consider an offset (radius, actually) of bin_offset
	__host__ __device__ INT GetBin(const TYPE discretization, const int16_t bin_offset, const gridPoint& lowest_node, const UINT PtsPerDimension) {
		INT bin_idx = 0;
 
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			INT temp_idx = (INT)roundf((dim[d] - lowest_node.dim[d]) / discretization) + bin_offset;
			bin_idx += temp_idx * powf(PtsPerDimension, d);
		}
		return bin_idx;
	};
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
	UINT node, AMR_selected;

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
__host__ __device__ 
inline UINT positive_rem(const INT a, const INT b){
	return (a % b + b) % b;
} 
// ------------------------------------------------------------------------ //
// ------------------------------------------------------------------------ //

__host__ __device__ inline Param_vec _Gather_Param_Vec(const UINT index, const Param_pair* Parameter_Array, const INT* n_Samples){

	Param_vec Output;

	Output.Joint_PDF = 1;

	UINT aux_samples_mult = 1;
	UINT aux_samples_sum  = 0;

	 
	for (uint16_t d = 0; d < PARAM_DIMENSIONS; d++){
		UINT aux3 = n_Samples[d];
		UINT aux = floorf(positive_rem(index, aux3 * aux_samples_mult) / aux_samples_mult );

		Output.sample_vec[d] = Parameter_Array[aux + aux_samples_sum].sample;
		Output.Joint_PDF 	*= Parameter_Array[aux + aux_samples_sum].PDF;

		aux_samples_mult *= aux3;
		aux_samples_sum  += aux3;
	}
	return Output;
}

__device__ __forceinline__ void __atomicAdd(double *address, double val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    uint64_t *ptr = (uint64_t *)address;
    uint64_t old, newdbl, ret = *ptr;
    do {
        old = ret;
        newdbl = __double_as_longlong(__longlong_as_double(old)+val);
    } while((ret = atomicCAS(ptr, old, newdbl)) != old);
}

__device__ __forceinline__ void __atomicAdd(float *address, float val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    UINT *ptr = (UINT *)address;
    UINT old, newint, ret = *ptr;
    do {
        old = ret;
        newint = __float_as_int(__int_as_float(old)+val);
    } while((ret = atomicCAS(ptr, old, newint)) != old);
}
// +===========================================================================+ //
// +===========================================================================+ //
// +===========================================================================+ //
// +===========================================================================+ //

__device__ inline gridPoint VECTOR_FIELD(gridPoint X, TYPE t, const Param_vec parameter, const UINT mode, const FIXED_TYPE extra_param[]) {
	gridPoint output = VEC_FIELD;

	return output;
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__ inline TYPE DIVERGENCE_FIELD(gridPoint X, TYPE t, const Param_vec parameter, const UINT mode, const FIXED_TYPE extra_param[]) {
	TYPE output = DIVERGENCE;

	return output;
}


#endif