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

/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a This is the numerator
/// @param b This is the denominator
/// @return Returns mod(a,b)
__host__ __device__
inline UINT positive_rem(const INT a, const INT b) {
	return (a % b + b) % b;
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

	// Default constructor	
	// __host__ __device__
	// gridPoint(){
	// 	for (uint16_t d = 0; d < DIMENSIONS; d++){
	// 		this->dim[d] = 0;
	// 	}
	// }

	__host__ __device__ 
	gridPoint operator+(const gridPoint& other) const {

		gridPoint out;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			TYPE aux = dim[d];
			aux += other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ 
	gridPoint operator-(const gridPoint& other) const {
		gridPoint out;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			TYPE aux = dim[d];
			aux -= other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ 
	bool operator==(const gridPoint& other) const {
		bool out = true;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			if (dim[d] != other.dim[d]) { out = false; }
		}

		return out;
	}

	__host__ __device__ 
	inline TYPE Distance(const gridPoint& other) const {
		TYPE dist = 0;
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			dist += (dim[d] - other.dim[d]) * (dim[d] - other.dim[d]);
		}
		return sqrtf(dist);
	};

	
	__host__ __device__ 
	inline gridPoint Mult_by_Scalar(TYPE scalar) const {
		gridPoint out;

		
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			out.dim[d] = scalar * dim[d];
		}

		return out;
	};

	// This function tells us what is the first bin to search if we consider an offset (radius, actually) of bin_offset
	__host__ __device__ 
	INT GetBin(const TYPE discretization, const int16_t bin_offset, const gridPoint& lowest_node, const UINT PtsPerDimension) const {
		INT bin_idx = 0;
 
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			INT temp_idx = (INT)roundf((dim[d] - lowest_node.dim[d]) / discretization) + bin_offset;
			bin_idx += temp_idx * powf(PtsPerDimension, d);
		}
		return bin_idx;
	};
};

class grid{
public:
	gridPoint 	Boundary_inf, Boundary_sup;
	UINT		Nodes_per_Dim;

	// Default constructor:
	__host__ __device__
	grid(){
		Boundary_inf	= DOMAIN_CTR;
		Boundary_sup	= DOMAIN_CTR;
		Nodes_per_Dim	= 0;
	}
	// Custom constructor:
	__host__ __device__
	grid(const gridPoint Center, const gridPoint Diameter, const INT Nodes_per_dim){

		Nodes_per_Dim	= Nodes_per_dim;
		Boundary_inf	= Center - Diameter.Mult_by_Scalar(0.5);
		Boundary_sup	= Center + Diameter.Mult_by_Scalar(0.5);
	}

// Methods/functions
	// This function gives the total amount of Mesh nodes
	__host__ __device__
	inline UINT Total_Nodes() const {
		return pow(Nodes_per_Dim, DIMENSIONS);
	}

	// This function gives the length of each edge of our cubical mesh
	__host__ __device__
	inline gridPoint Edge_size() const {
		return (Boundary_sup - Boundary_inf);
	}

	// This function returns the center point of the mesh
	__host__ __device__
	inline gridPoint Center() const {
		return (Boundary_sup + Boundary_inf).Mult_by_Scalar(0.5);
	}

	// This function gives the mesh discretization length
	__host__ __device__
		inline TYPE Discr_length() const {
		if (Nodes_per_Dim == 1) { return (TYPE)0; }

		return (TYPE)(this->Edge_size().dim[0] / (Nodes_per_Dim - 1));
	}

	// Given an index, this function returns the corresponding node
	__host__ __device__
	inline gridPoint Get_node(const INT& idx) const {
		gridPoint out;

		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			INT j = floor( positive_rem(idx, pow(Nodes_per_Dim, d + 1)) / pow(Nodes_per_Dim, d) );		// This line gives the index at each dimension
			out.dim[d] = ((TYPE) j / (Nodes_per_Dim - 1) - 0.50f) * this->Edge_size().dim[d]; // This line gives the grid node per se
		}
		out = out + this->Center();

		return out;
	}

	// Checks whether Particle belongs to the grid/mesh
	__host__ __device__
	inline bool Contains_particle(const gridPoint Particle) const {
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			if (Particle.dim[d] < this->Boundary_inf.dim[d] || Particle.dim[d] > this->Boundary_sup.dim[d]) { return false; }
		}
		return true;
	}
	
	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	__host__ __device__ 
	inline INT Give_Bin(const gridPoint Particle, const INT bin_offset) const {
		INT bin_idx = 0;
 
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			INT temp_idx = roundf((Particle.dim[d] - this->Boundary_inf.dim[d]) / this->Discr_length()) + bin_offset;
			bin_idx += temp_idx * powf(Nodes_per_Dim, d);
		}
		return bin_idx;
	};

	// Compute the global index in a mesh, given the global index in another mesh.
	__host__ __device__
	inline INT Indx_here(const INT indx_at_other, const grid& other) const {

		INT out = this->Give_Bin(other.Boundary_inf,0);

		for (uint16_t d = 0; d < DIMENSIONS; d++) {

			// Calculate the local dimensional index in the 'other' mesh:
			out += floor(positive_rem(indx_at_other, pow(other.Nodes_per_Dim, d + 1)) / pow(other.Nodes_per_Dim, d)) * pow(this->Nodes_per_Dim, d);
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