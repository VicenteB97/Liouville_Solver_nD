#include <Adapt_Mesh.cuh>

/// @brief (DEVICE FUNCTION) Compute a 1D Haar wavelet transform
/// @param s1 
/// @param s2 
/// @return 
__device__ inline void _1D_WVLET(floatType& s1, floatType& s2) {

	floatType aux = 0.5 * (s1 + s2);
	s2 = s1 - s2;
	s1 = aux;
}

template<uint16_t elementsProcessedPerThread>
__global__ void customAssignToGpuArray(Particle* outputNodes, const Mesh inputNodes,
	const uintType* nodeIdx, const intType elementNr) {
	const int64_t globalId = blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
	for (uint16_t k = 0; k < elementsProcessedPerThread; k++) {

		const uintType myIdx = globalId * elementsProcessedPerThread + k;

		if (myIdx < elementNr) {
			const intType myNodeIdx = nodeIdx[myIdx];
			outputNodes[myIdx] = inputNodes.Get_node(myNodeIdx);
		}
	}
}

__global__
void write_signal_in_bounding_box(
	const floatType* input_signal,
	floatType* output_signal,
	const Mesh signal_domain,
	const Mesh signal_bounding_box,
	const uint64_t max_nodes
){
	const uint64_t global_id = blockDim.x * blockIdx.x + threadIdx.x;

	if(global_id >= max_nodes){ return; }

	// First, find the node index at the problem domain:
	uint64_t node_index_at_domain = signal_domain.Get_binIdx(signal_bounding_box.Get_node(global_id));

	// Now, fill the value of the output_signal with the value of the input_signal
	output_signal[global_id] = input_signal[node_index_at_domain];
}

__host__
thrust::device_vector<floatType> setInitialParticles(
	const thrust::device_vector<floatType>& input_signal_gpu,
	thrust::device_vector<Particle>& output_active_nodes,
	const Mesh& signal_bounding_box,
	const Mesh& signal_domain
){
	// Create the signal in the bounding box. Initialized to 0
	thrust::device_vector<floatType> signal_in_bounding_box(signal_bounding_box.size(), (floatType) 0);
	
	// Fill the signal_in_bounding_box
	uint16_t threads = fmin(THREADS_P_BLK, signal_in_bounding_box.size());
	uint64_t blocks = floor((signal_in_bounding_box.size() - 1) / threads ) + 1;

	write_signal_in_bounding_box <<<threads, blocks>>> (
		rpc(input_signal_gpu, 0),
		rpc(signal_in_bounding_box, 0),
		signal_domain,
		signal_bounding_box,
		signal_in_bounding_box.size()
	);

	// Create amr_handle
	waveletMeshRefinement_GPU amr_handle;

	amr_handle.set_min_refinement_level() = 0;
	amr_handle.set_max_refinement_level() = log2(signal_bounding_box.Nodes_per_Dim);
	amr_handle.set_initial_signal(signal_in_bounding_box);

	amr_handle.compute_wavelet_transform();
	amr_handle.get_detail_above_threshold_nodes(output_active_nodes, signal_bounding_box);

	return amr_handle.transformed_signal()
}

__host__ __device__
waveletMeshRefinement_GPU::waveletMeshRefinement_GPU(){};
waveletMeshRefinement_GPU::~waveletMeshRefinement_GPU(){ delete this;};

__host__ __device__
void waveletMeshRefinement_GPU::set_min_refinement_level(uint16_t input){
	__min_refinement_level = input;
};

__host__ __device__
uint16_t waveletMeshRefinement_GPU::min_refinement_level() const {
	return __min_refinement_level;
};

__host__ __device__
void waveletMeshRefinement_GPU::set_max_refinement_level(uint16_t input){
	__max_refinement_level = input;
};

__host__ __device__
uint16_t waveletMeshRefinement_GPU::max_refinement_level() const {
	return __max_refinement_level;
};

__host__
void waveletMeshRefinement_GPU::set_initial_signal(const thrust::device_vector<floatType>& input_signal){
	__initial_signal = input_signal;
};

__host__ __device__
uint32_t waveletMeshRefinement_GPU::nodes_per_dimension() const {
	return pow((double) __initial_signal.size(), (double) 1/DIM);
};

__host__ __device__
void waveletMeshRefinement_GPU::total_nodes() const {
	return __initial_signal.size();
};

__host__
thrust::device_vector<floatType> waveletMeshRefinement_GPU::initial_signal() const {
	return __initial_signal;
};

__host__
thrust::device_vector<floatType> waveletMeshRefinement_GPU::transformed_signal() const {
	return __transformed_signal;
};	// Include a copy from the thrust container to the initial type

__host__
thrust::device_vector<uint64_t> waveletMeshRefinement_GPU::assigned_node_indeces() const {
	return __assigned_node_indeces;
};

__host__
thrust::device_vector<uint64_t> waveletMeshRefinement_GPU::assigned_node_marker() const {
	return __assigned_node_marker;
};

__host__
void waveletMeshRefinement_GPU::compute_wavelet_transform(){
	// Here, you've got to compute the wavelet transformation of the initial signal.
	// Pass the wavelet transform as a function pointer (future)
	__transformed_signal = __initial_signal;

	uint32_t rescaling {2};
	const uint64_t total_nodes {this->total_nodes()};
	const uint32_t nodes_per_dimension {this->nodes_per_dimension()};
	const double tolerance {1E-5}; 

	for (uint16_t k = 0; k < __max_refinement_level - __min_refinement_level + 1; k++) {

		uint16_t Threads = fmin(THREADS_P_BLK, total_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS));
		uint64_t Blocks = floor((total_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS) - 1) / Threads) + 1;

		single_block_single_level_wt_GPU<< <Blocks, Threads >> > (
			rpc(__transformed_signal, 0), rescaling, nodes_per_dimension, total_nodes
		);
		gpuError_Check(cudaDeviceSynchronize());

		get_nodes_above_threshold<< <Blocks, Threads >> > (
			rpc(__transformed_signal, 0), 
			rpc(__assigned_node_indeces, 0), 
			rpc(__assigned_node_markers, 0), 
			rescaling, nodes_per_dimension, total_nodes, tolerance
		);
		gpuError_Check(cudaDeviceSynchronize());

		rescaling *= 2;	// our Mesh will now have half the number of points
	}
};

__host__
void waveletMeshRefinement_GPU::get_detail_above_threshold_nodes(
	thrust::device_vector<Particle>& particle_locations, 
	const Mesh& signal_domain
) const {
	// Here we assume that the compute_wavelet transform function has already been called
	// Get the number of assigned nodes
	const uintType nrSelectedNodes = thrust::reduce(thrust::device, __assigned_node_marker.begin(), __assigned_node_marker.end());

	// Set the selected nodes first
	thrust::sort_by_key(
		thrust::device, 
		__assigned_node_marker.begin(), 
		__assigned_node_marker.end(), 
		__assigned_node_indeces.begin(), 
		thrust::greater<intType>()
	);

	particle_locations.resize(nrSelectedNodes);

	const intType Threads = fmin(THREADS_P_BLK, nrSelectedNodes);
	const intType Blocks = floor((nrSelectedNodes - 1) / Threads) + 1;

	const uint16_t elementsAtATime = ELEMENTS_AT_A_TIME;

	customAssignToGpuArray<elementsAtATime> << <Threads, Blocks >> > (rpc(particle_locations, 0),
		signal_domain, rpc(__assigned_node_indeces, 0), nrSelectedNodes);
	gpuError_Check(cudaDeviceSynchronize());
};


// Put signal dimensions as input parameter
// template<typename floatType, uint16_t PHASE_SPACE_DIMENSIONS>
__global__ inline void single_block_single_level_wt_GPU(
	floatType* signal,
	const floatType rescaling,
	const uint32_t nodes_per_dimension,
	const uint64_t total_nodes
){
	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// Range guard for out-of-bounds nodes
	if (globalID >= total_nodes / powf(rescaling, PHASE_SPACE_DIMENSIONS)) { return; }
	
	// Total nodes in each simple wavelet transform (per GPU thread)
	const uint16_t	miniSquareNodes = pow(2, PHASE_SPACE_DIMENSIONS);				

	// Global index of the main approximation vertex at the bounding box
	int64_t cube_app_IDX = 0;

	// Compute the lowest corner node index
	uint64_t multCounter = 1;	// auxiliary counter: => pow(BoundingBox.Nodes_per_Dim / rescaling, d)
	uint64_t multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		int64_t temp_idx = floorf(positive_rem(globalID, multCounter * (total_nodes / rescaling)) / multCounter) * rescaling;

		cube_app_IDX += temp_idx * multCounter_2;
		multCounter *= total_nodes / rescaling;
		multCounter_2 *= total_nodes;
	}

	multCounter = 1;	// Reinitialize for next computations: => pow(2, d)
	multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)

	// 1 set of wavelets per dimension (1D: horizontal; 2D: Horizontal + Vertical; 3D: Horz + Vert + Deep; ...)
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		// Go through all the vertices that are defined by the main cube approximation vertex
		for (uint64_t k = 0; k < miniSquareNodes; k++) {
			// If we are at the current approximation vertex:
			if (floorf(positive_rem(k, 2 * multCounter) / multCounter) != 0) { continue; }
			// here, multCounter == pow(2, d)

			// Compute approximation node
			uint64_t app_IDX_at_BBox = cube_app_IDX;

			uint64_t multCounter_3 = 1;	// => pow(2, j)
			uint64_t multCounter_4 = 1;	// => pow(BoundingBox.Nodes_per_Dim, j)

			for (uint16_t j = 0; j < PHASE_SPACE_DIMENSIONS; j++) {
				int64_t temp = floorf(positive_rem(k, multCounter_3 * 2) / multCounter_3) * rescaling / 2;	// j-th component index

				app_IDX_at_BBox += temp * multCounter_4;
				multCounter_3 *= 2;
				multCounter_4 *= total_nodes;
			}
			// Compute corresponding detail node
			int64_t det_IDX_at_BBox = app_IDX_at_BBox + multCounter_2 * rescaling / 2;
			// Compute the wavelet transform in-place
			_1D_WVLET(signal[app_IDX_at_BBox], signal[det_IDX_at_BBox]);
		}
		multCounter *= 2;
		multCounter_2 *= nodes_per_dimension;
	}
}


// template<typename floatType, uint16_t PHASE_SPACE_DIMENSIONS>
__global__ 
inline void get_nodes_above_threshold(
	floatType* signal,
	uint64_t* assigned_node_indeces,
	uint32_t* assigned_node_markers,
	const floatType rescaling,
	const uint64_t nodes_per_dimension,
	const uint64_t total_nodes,
	const double tolerance
)
{
	const uint64_t global_id = blockDim.x * blockIdx.x + threadIdx.x;
	if(global_id >= total_nodes / powf(rescaling, PHASE_SPACE_DIMENSIONS)){
		return;
	}
	// Total nodes in the problem domain
	const uint16_t	miniSquareNodes = pow(2, PHASE_SPACE_DIMENSIONS);

	// Global index of the main approximation vertex at the bounding box
	int64_t cube_app_IDX = 0;

	uint64_t multCounter = 1;	// auxiliary counter: => pow(BoundingBox.Nodes_per_Dim / rescaling, d)
	uint64_t multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		int64_t temp_idx = floorf(
			positive_rem(globalID, multCounter * (nodes_per_dimension / rescaling)) / multCounter
		) * rescaling;

		cube_app_IDX += temp_idx * multCounter_2;
		multCounter *= nodes_per_dimension / rescaling;
		multCounter_2 *= nodes_per_dimension;
	}

	// Now we have to go see what happens with the outputs
	nodeIdxs[cube_app_IDX] = 0;

	for (uint64_t k = 1; k < miniSquareNodes; k++) {
		// Particle visit_node(BoundingBox.Get_node(cube_app_IDX));
		uint64_t detail_idx = cube_app_IDX;

		multCounter = 1;
		multCounter_2 = 1;

		// Get the indeces at the bounding box:
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			uint64_t temp = floorf(positive_rem(k, multCounter * 2) / multCounter) * rescaling / 2;	// j-th component index

			detail_idx += temp * multCounter_2;
			multCounter *= 2;
			multCounter_2 *= total_nodes;
		}

		assigned_node_indeces[detail_idx] = detail_idx;

		if (abs(signal[detail_idx]) >= tolerance) {
			assigned_node_markers[detail_idx] = 1;
		}
	}
}
