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

#define double double

#define MAX_FILE_SIZE_B (uint64_t)2*1024*1024*1024

#if DIMENSIONS < 3
	#define INT int32_t
	#define UINT uint32_t
#else
	#define INT int64_t
	#define UINT uint64_t
#endif

//Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

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
#define ptSEARCH_THRESHOLD 20000 // The min. number of particles per sample where we use the counting sort search

//-------------------------------------------------------------------------//
//-------------------------- CLASSES USED ---------------------------------//
//-------------------------------------------------------------------------//
// Grid points----------------------------------------------------
template<uint16_t DIM, class T>
class gridPoint {
public:
	T dim[DIM];

	// Default constructor
	__host__ __device__
	gridPoint<DIM, T>() {
		for (uint16_t d = 0; d < DIM; d++) {
			dim[d] = (T)0;
		}
	}

	// Parametric constructors
	__host__ __device__
		gridPoint<DIM, T>(const T(&input)[DIM]) {
		for (uint16_t d = 0; d < DIM; d++) {
			dim[d] = input[d];
		}
	}

	__host__ __device__
	gridPoint<DIM, T>(const gridPoint<DIM,T>& input) {
		for (uint16_t d = 0; d < DIM; d++) {
			dim[d] = input.dim[d];
		}
	}

	// Operators and methods
	__host__ __device__ 
	gridPoint<DIM, T> operator+(const gridPoint<DIM, T>& other) const {

		gridPoint<DIM, T> out;

		for (uint16_t d = 0; d < DIM; d++) {
			T aux = dim[d];
			aux += other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ 
	gridPoint<DIM, T> operator-(const gridPoint<DIM, T>& other) const {
		gridPoint<DIM, T> out;

		for (uint16_t d = 0; d < DIM; d++) {
			T aux = dim[d];
			aux -= other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ 
	bool operator==(const gridPoint<DIM, T>& other) const {
		bool out = true;
		for (uint16_t d = 0; d < DIM; d++) {
			if (dim[d] != other.dim[d]) { out = false; }
		}
		return out;
	}

	__host__ __device__ 
	inline T Distance(const gridPoint<DIM, T>& other) const {
		T dist = 0;
		for (uint16_t d = 0; d < DIM; d++) {
			dist += (dim[d] - other.dim[d]) * (dim[d] - other.dim[d]);
		}
		return sqrtf(dist);
	};

	
	__host__ __device__ 
	inline gridPoint<DIM, T> Mult_by_Scalar(T scalar) const {
		gridPoint<DIM, T> out;

		
		for (uint16_t d = 0; d < DIM; d++) {
			out.dim[d] = scalar * dim[d];
		}

		return out;
	};
};

template<uint16_t DIM, class T>
class grid{
public:
	gridPoint<DIM, T> 	Boundary_inf, Boundary_sup;
	UINT				Nodes_per_Dim;

	// Default constructor:
	__host__ __device__	grid<DIM, T>(){
		Nodes_per_Dim	= 1;
		Boundary_inf	= gridPoint<DIM, T>(DOMAIN_INF);
		Boundary_sup	= gridPoint<DIM, T>(DOMAIN_SUP);
	}

	// Parametric constructors:
	__host__ __device__	grid<DIM, T>(const INT& Nodes_per_dim) {

		Nodes_per_Dim = Nodes_per_dim;
		Boundary_inf  = gridPoint<DIM, T>(DOMAIN_INF);
		Boundary_sup  = gridPoint<DIM, T>(DOMAIN_SUP);
	}

	__host__ __device__	grid<DIM, T>(const T& Discretization_length) {

		Boundary_inf = gridPoint<DIM, T>(DOMAIN_INF);
		Boundary_sup = gridPoint<DIM, T>(DOMAIN_SUP);
		Nodes_per_Dim = ceil((Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Discretization_length);
	}

	__host__ __device__	grid<DIM, T>(const gridPoint<DIM, T>& Bnd_inf, const gridPoint<DIM, T>& Bnd_sup) {

		Nodes_per_Dim = 2;
		Boundary_inf = Bnd_inf;
		Boundary_sup = Bnd_sup;
	}

	__host__ __device__	grid<DIM, T>(const gridPoint<DIM, T>& Bnd_inf, const gridPoint<DIM, T>& Bnd_sup, const INT& Nodes_per_dim){

		Nodes_per_Dim	= Nodes_per_dim;
		Boundary_inf	= Bnd_inf;
		Boundary_sup	= Bnd_sup;
	}

	__host__ __device__	grid<DIM, T>(const gridPoint<DIM, T>& Bnd_inf, const gridPoint<DIM, T>& Bnd_sup, const T& Discretization_length) {

		Boundary_inf = Bnd_inf;
		Boundary_sup = Bnd_sup;
		Nodes_per_Dim = ceil((Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Discretization_length);
	}

// Methods/functions
	// This function gives the total amount of Mesh nodes
	__host__ __device__	inline UINT Total_Nodes() const {
		return pow(Nodes_per_Dim, DIM);
	}

	// This function gives the length of each edge of our cubical mesh
	__host__ __device__	inline T Edge_size() const {
		return (Boundary_sup.dim[0] - Boundary_inf.dim[0]);
	}

	// This function returns the center point of the mesh
	__host__ __device__	inline gridPoint<DIM, T> Center() const {
		return (Boundary_sup + Boundary_inf).Mult_by_Scalar(0.5);
	}

	// This function gives the mesh discretization length
	__host__ __device__	inline TYPE Discr_length() const {
		if (Nodes_per_Dim == 1) { return (TYPE)0; }

		return (TYPE)(this->Edge_size() / (Nodes_per_Dim - 1));
	}

	// Given an index, this function returns the corresponding node
	__host__ __device__	inline gridPoint<DIM, T> Get_node(const INT& globalIdx) const {

		gridPoint<DIM, T> out;

		for (uint16_t d = 0; d < DIM; d++) {
			INT j = floor( positive_rem(globalIdx, pow(Nodes_per_Dim, d + 1)) / pow(Nodes_per_Dim, d) );	// This line gives the index at each dimension

			out.dim[d] = j * this->Discr_length() + Boundary_inf.dim[d];									// This line gives the grid<DIM, TYPE> node per se
		}
		return out;
	}

	// Checks whether Particle belongs to the INTERIOR+BOUNDARY of the grid/mesh
	__host__ __device__	inline bool Contains_particle(const gridPoint<DIM, T>& Particle) const {
		for (uint16_t d = 0; d < DIM; d++) {
			if (Particle.dim[d] < Boundary_inf.dim[d] || Particle.dim[d] > Boundary_sup.dim[d]) { return false; }
		}
		return true;
	}
	
	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	__host__ __device__ inline INT Get_binIdx(const gridPoint<DIM, T>& Particle, const INT& bin_offset) const {
		INT bin_idx = 0;
 
		for (uint16_t d = 0; d < DIM; d++) {
			INT temp_idx = roundf((Particle.dim[d] - Boundary_inf.dim[d]) / this->Discr_length()) + bin_offset;

			bin_idx += temp_idx * powf(Nodes_per_Dim, d);
		}
		return bin_idx;
	};

	// Compute the global index at your mesh, given the global index in "other" mesh.
	__host__ __device__	inline INT Indx_here(const INT& indx_at_other, const grid<DIM, T>& other) const {
		return this->Get_binIdx(other.Get_node(indx_at_other),0);
	}

	/// @brief This function expands a fixed grid "Other" by a length of  "expansion_length" in each direction/dimension
	__host__ __device__	inline void Expand_From(const grid& Other, const T& expansion_length) {
		
		for (uint16_t d = 0; d < DIM; d++) {
			// To make sure that the points fall into the grid nodes
			Boundary_inf.dim[d] = Other.Boundary_inf.dim[d] - Other.Discr_length() * floorf(expansion_length / Other.Discr_length());
			Boundary_sup.dim[d] = Other.Boundary_sup.dim[d] + Other.Discr_length() * ceilf( expansion_length / Other.Discr_length());
		}

		Nodes_per_Dim = ceil((Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Other.Discr_length());
	}

	/// @brief This function makes you domain a square (same Lebesgue-length in every direction)
	__host__ __device__ inline void Squarify() {
		// Get the max distance between the edges and then make the box larger!
		T max_dist = Boundary_sup.dim[0] - Boundary_inf.dim[0];

		for (uint16_t d = 1; d < DIM; d++) {
			max_dist = fmax(max_dist, Boundary_sup.dim[d] - Boundary_inf.dim[d]);
		}

		// Now that we know the max dist, let's expand the edges!
		for (uint16_t d = 0; d < DIM; d++) {
			Boundary_sup.dim[d] = Boundary_inf.dim[d] + max_dist;
		}
	}
};

// Time + impulse: ----------------------------------------------
class Time_Impulse_vec {
public:
	double 	time;
	bool 		impulse;

	bool operator < (const Time_Impulse_vec& other) const {
		return (time < other.time);
	}
};

class AMR_node_select{
public:
	UINT node, AMR_selected;

	__host__ __device__ 
	bool operator < (const AMR_node_select& other) const { // Note that we have changed order, for simpler work...
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
	TYPE params[2];				// mean and variance. The appropriate choice of distribution parameters are given by the method of moments
	char Name;					// Distribution name. Currently supported distributions: Delta, Normal, Beta, Gamma and Uniform
	bool Truncated;				// Truncated? TRUE or FALSE
	TYPE trunc_interval[2];		// Truncation interval (give min and max of the interval)

	// Default constructor
	Distributions() {
		params[0] = 0;
		params[1] = 0;

		Name = 'D';

		Truncated = true;

		trunc_interval[0] = 0;
		trunc_interval[1] = 0;
	}
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
template<uint16_t DIM, class T>
__device__ inline gridPoint<DIM, T> VECTOR_FIELD(gridPoint<DIM, T> X, T t, const Param_vec parameter, const UINT mode, const double extra_param[]) {

	return { VEC_FIELD };
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
template<uint16_t DIM, class T>
__device__ inline T DIVERGENCE_FIELD(gridPoint<DIM, T> X, T t, const Param_vec parameter, const UINT mode, const double extra_param[]) {

	return DIVERGENCE;
}


#endif