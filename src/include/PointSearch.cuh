#ifndef __POINTSEARCH_CUH__
#define __POINTSEARCH_CUH__

#include "Case_definition.cuh"
#include "headers.cuh"
#include "utils/error_functs.cuh"
#include "utils/others.cuh"
#include "utils/numeric_defs.cuh"
#include "Domain.cuh"
#include "Interpolation.cuh"

#if 0
/// @brief 
/// @param Bin_locations 
/// @param Bin_count 
/// @param Bin_local_accSum 
/// @param Search_Particles 
/// @param Bounding_Box 
/// @param Total_Particles 
/// @return 
__global__ void Bin_Insertion_Count(
	uintType* Bin_locations,
	uintType* Bin_count,
	uintType* Bin_local_accSum,
	const Particle* Search_Particles,
	const Mesh	Bounding_Box,
	const uintType	Total_Particles);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


				/// @brief 
/// @param fixed_Particles 
/// @param Particles_2sort 
/// @param fixed_values 
/// @param values_2sort 
/// @param Bin_count 
/// @param Bin_locations 
/// @param Bin_local_accSum 
/// @param Total_Particles 
/// @param offset 
/// @return 
__global__ void Count_sort(
	const Particle* fixed_Particles,
	Particle* Particles_2sort,
	const floatType* fixed_values,
	floatType* values_2sort,
	const uintType* Bin_count,
	const uintType* Bin_locations,
	const uintType* Bin_local_accSum,
	const uintType				Total_Particles,
	const intType				offset);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 
/// @param Search_Particles 
/// @param Bin_count 
/// @param Index_Array 
/// @param Matrix_Entries 
/// @param Num_Neighbors 
/// @param max_neighbor_num 
/// @param Total_Particles 
/// @param offset 
/// @param search_distance 
/// @param Bounding_Box 
/// @return 
__global__ void Neighbor_search(
	Particle* Search_Particles,
	const uintType* Bin_count,
	intType* Index_Array,
	floatType* Matrix_Entries,
	const uintType	max_neighbor_num,
	const uintType	Total_Particles,
	const uintType	offset,
	const floatType			search_distance,			// This tells us how many discretizations we have to move back to find initial bin to search from
	const Mesh		Bounding_Box);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Counting sort! This operation orders the Mesh indeces and it sets the number of nodes inside each bin

/// @brief 
/// @tparam floatType 
/// @tparam PHASE_SPACE_DIMENSIONS 
/// @param Search_Particles 
/// @param PDF_vals 
/// @param Index_Array 
/// @param Matrix_Entries 
/// @param Num_Neighbors 
/// @param Adapt_Points 
/// @param max_neighbor_num 
/// @param Bounding_Box 
/// @param search_radius 
/// @return 
__host__ int16_t CS_Neighbor_Search(thrust::device_vector<Particle>& Search_Particles,
	thrust::device_vector<floatType>& PDF_vals,
	thrust::device_vector<intType>& Index_Array,
	thrust::device_vector<floatType>& Matrix_Entries,
	const uintType Adapt_Points,
	const uintType max_neighbor_num,
	const Mesh& Bounding_Box,
	const floatType search_radius);

#endif // 0


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/// @brief 
/// @tparam PHASE_SPACE_DIMENSIONS
/// @tparam floatType
/// @param Search_Particles 
/// @param Fixed_Particles 
/// @param Index_Array 
/// @param Matrix_Entries
/// @param max_neighbor_num 
/// @param Adapt_Points 
/// @param Total_Particles 
/// @param search_radius 
/// @return 
__global__ void Exh_PP_Search(const Particle* Search_Particles,
	const Particle* Fixed_Particles,
	intType* Index_Array,
	floatType* Matrix_Entries,
	const intType max_neighbor_num,
	const uintType Adapt_Points,
	const uintType Total_Particles,
	const floatType search_radius);


int16_t particleNeighborSearch(thrust::device_vector<Particle>& Search_Particles,
	thrust::device_vector<floatType>& PDF_vals,
	thrust::device_vector<intType>& Index_Array,
	thrust::device_vector<floatType>& Matrix_Entries,
	const uintType Adapt_Points,
	const uintType MaxNeighborNum,
	const Mesh& Bounding_Box,
	const floatType search_radius);

#endif
