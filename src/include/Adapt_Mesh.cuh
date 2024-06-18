#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "headers.cuh"
#include "utils/error_functs.cuh"
#include "utils/numeric_defs.cuh"
#include "utils/others.cuh"
#include "Domain.cuh"
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename floatType, uint16_t PHASE_SPACE_DIMENSIONS>
class waveletMeshRefinement_GPU{
public:
	// Default constructor and destrucor
	waveletMeshRefinement_GPU(){};
	~waveletMeshRefinement_GPU(){};

private:
	// Input signal for wavelet transform: 
	thrust::device_vector<floatType> __initial_signal;

	// Parameters
	uint16_t __min_refinement_level; 
	uint16_t __max_refinement_level;
	bool __is_transform_in_place = true;

	// Output signals
	thrust::device_vector<floatType> __transformed_signal;
	thrust::device_vector<floatType> __threshold_cutoff_transformed_signal;

	// Output AMR arrays for indeces and assigned nodes
	thrust::device_vector<uint64_t> __assigned_node_indeces;
	thrust::device_vector<uint32_t> __assigned_node_markers;

public:
	__host__ __device__
	void set_min_refinement_level(uint16_t input);

	__host__ __device__
	uint16_t min_refinement_level() const;
	
	__host__ __device__
	void set_max_refinement_level(uint16_t input);

	__host__ __device__
	uint16_t max_refinement_level() const;

	__host__
	void set_initial_signal(const thrust::device_vector<floatType>& input_signal);

	__host__ __device__
	uint32_t nodes_per_dimension() const;

	__host__ __device__
	void total_nodes() const;

	__host__
	thrust::device_vector<floatType> initial_signal() const;

	__host__
	thrust::device_vector<floatType> transformed_signal() const;

	__host__
	thrust::device_vector<uint64_t> assigned_node_indeces() const;
	
	__host__
	thrust::device_vector<uint64_t> assigned_node_marker() const;

	__host__
	void compute_wavelet_transform();

	__host__
	void get_detail_above_threshold_nodes(
		thrust::device_vector<Particle>& particle_locations, 
		const Mesh& problem_domain
	) const;
}

// Put signal dimensions as input parameter
// template<typename floatType, uint16_t PHASE_SPACE_DIMENSIONS>
__global__ inline void single_block_single_level_wt_GPU(
	floatType* signal,
	const floatType rescaling,
	const uint32_t nodes_per_dimension,
	const uint64_t total_nodes
);


// template<typename floatType, uint16_t PHASE_SPACE_DIMENSIONS>
__global__ inline void get_detail_above_threshold_nodes(
	floatType* signal,
	uint64_t* assigned_node_indeces,
	uint32_t* assigned_node_markers,
	const floatType rescaling,
	const uint64_t nodes_per_dimension,
	const uint64_t total_nodes,
	const double tolerance
);


/// @brief (DEVICE FUNCTION) Compute a 1D Haar wavelet transform
/// @param s1 
/// @param s2 
/// @return 
__device__ inline void _1D_WVLET(floatType& s1, floatType& s2);


template<uint16_t elementsProcessedPerThread>
__global__ void customAssignToGpuArray(
	Particle* outputNodes, 
	const Mesh inputNodes,
	const uint64_t* nodeIdx, 
	const intType elementNr
);

__host__
thrust::device_vector<floatType> setInitialParticles(
	thrust::device_vector<floatType>& input_signal_gpu,
	thrust::device_vector<Particle>& output_active_nodes,
	const Mesh& signal_bounding_box,
	const Mesh& signal_domain
);

__global__
void write_signal_in_bounding_box(
	const floatType* input_signal,
	floatType* output_signal,
	const Mesh signal_domain,
	const Mesh signal_bounding_box,
	const uint64_t max_nodes
);

/// @brief (GLOBAL FUNCTION) Compute 1 level of the multidimensional wavelet transform in the GPU
/// @tparam PHASE_SPACE_DIMENSIONS
/// @tparam floatType
/// @param PDF Our "signal". The multidimensional signal we want to compress
/// @param Activate_node An array with the nodes and the indication whether the node is chosen or not
/// @param BoundingBox The "smallest" Mesh where the support of the PDF is contained
/// @param Problem_Domain Problem domain
/// @param rescaling Rescaling value that indicates the level of the wavelet transform
/// @return 
__global__ void D__Wavelet_Transform__F(
	floatType* PDF,
	uint64_t* nodeIdxs,
	uint64_t* isActiveNode,
	const Mesh 	BoundingBox,
	const Mesh	Problem_Domain,
	const floatType	rescaling
);

#endif