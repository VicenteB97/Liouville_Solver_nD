#pragma once

#include <cuda_wrappers.hpp>
#define deviceFunction __device__
#define hostFunction __host__
#define deviceClass cudaDevice
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief (DEVICE FUNCTION) Compute a 1D Haar wavelet transform
/// @param s1 
/// @param s2 
/// @return
template <typename floatType>
deviceFunction inline void haar_wavelet(floatType& s1, floatType& s2);

template<uint16_t elementsProcessedPerThread>
class customAssignToGpuArray {
public:
	Particle* outputNodes;
	const Mesh inputNodes;
	const uint64_t* nodeIdx;
	const uint32_t elementNr;
public:
	deviceFunction void operator()(const uint64_t global_id) const {
#pragma unroll
		for (uint16_t k = 0; k < elementsProcessedPerThread; k++) {

			const uintType myIdx = global_id * elementsProcessedPerThread + k;

			if (myIdx < elementNr) {
				const intType myNodeIdx = nodeIdx[myIdx];
				outputNodes[myIdx] = inputNodes.Get_node(myNodeIdx);
			}
		}
	};
};

template<typename floatType>
class single_block_single_level_wavelet_transform {
public:
	floatType* signal;
	uint64_t rescaling;
	uint32_t nodes_per_dimension;
	uint64_t total_signal_nodes;
	uint16_t dimensions;
public:
	deviceFunction void operator()(const uint64_t global_id) const {
		// Total nodes in each simple wavelet transform (per GPU thread)
		const uint16_t	miniSquareNodes = pow(2, dimensions);

		// Global index of the main approximation vertex at the bounding box
		int64_t cube_app_IDX = 0;

		// Compute the lowest corner node index
		uint64_t multCounter = 1;	// auxiliary counter: => pow(BoundingBox.Nodes_per_Dim / rescaling, d)
		uint64_t multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)
		for (uint16_t d = 0; d < dimensions; d++) {
			int64_t temp_idx = floorf(positive_rem(global_id, multCounter * (total_signal_nodes / rescaling)) / multCounter) * rescaling;

			cube_app_IDX += temp_idx * multCounter_2;
			multCounter *= total_signal_nodes / rescaling;
			multCounter_2 *= total_signal_nodes;
		}

		multCounter = 1;	// Reinitialize for next computations: => pow(2, d)
		multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)

		// 1 set of wavelets per dimension (1D: horizontal; 2D: Horizontal + Vertical; 3D: Horz + Vert + Deep; ...)
		for (uint16_t d = 0; d < dimensions; d++) {
			// Go through all the vertices that are defined by the main cube approximation vertex
			for (uint64_t k = 0; k < miniSquareNodes; k++) {
				// If we are at the current approximation vertex:
				if (floorf(positive_rem(k, 2 * multCounter) / multCounter) != 0) { continue; }
				// here, multCounter == pow(2, d)

				// Compute approximation node
				uint64_t app_IDX_at_BBox = cube_app_IDX;

				uint64_t multCounter_3 = 1;	// => pow(2, j)
				uint64_t multCounter_4 = 1;	// => pow(BoundingBox.Nodes_per_Dim, j)

				for (uint16_t j = 0; j < dimensions; j++) {
					int64_t temp = floorf(positive_rem(k, multCounter_3 * 2) / multCounter_3) * rescaling / 2;	// j-th component index

					app_IDX_at_BBox += temp * multCounter_4;
					multCounter_3 *= 2;
					multCounter_4 *= total_signal_nodes;
				}
				// Compute corresponding detail node
				int64_t det_IDX_at_BBox = app_IDX_at_BBox + multCounter_2 * rescaling / 2;
				// Compute the wavelet transform in-place
				haar_wavelet(signal[app_IDX_at_BBox], signal[det_IDX_at_BBox]);
			}
			multCounter *= 2;
			multCounter_2 *= nodes_per_dimension;
		}
	};
};


template<typename floatType>
class get_nodes_above_threshold {
public:
	floatType* signal;
	uint64_t* assigned_node_indeces;
	uint32_t* assigned_node_markers;
	const uint64_t rescaling;
	const uint64_t nodes_per_dimension;
	const uint64_t total_signal_nodes;
	const double tolerance;
	const uint16_t dimensions;
public:
	deviceFunction void operator()(const uint64_t global_id) const {
		if (global_id >= total_signal_nodes / powf(rescaling, dimensions)) {
			return;
		}
		// Total nodes in the problem domain
		const uint16_t	miniSquareNodes = pow(2, dimensions);

		// Global index of the main approximation vertex at the bounding box
		int64_t cube_app_IDX = 0;

		uint64_t multCounter = 1;	// auxiliary counter: => pow(BoundingBox.Nodes_per_Dim / rescaling, d)
		uint64_t multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)
		for (uint16_t d = 0; d < dimensions; d++) {
			int64_t temp_idx = floorf(
				positive_rem(global_id, multCounter * (nodes_per_dimension / rescaling)) / multCounter
			) * rescaling;

			cube_app_IDX += temp_idx * multCounter_2;
			multCounter *= nodes_per_dimension / rescaling;
			multCounter_2 *= nodes_per_dimension;
		}

		// Now we have to go see what happens with the outputs
		assigned_node_markers[cube_app_IDX] = 0;

		for (uint64_t k = 1; k < miniSquareNodes; k++) {
			// Particle visit_node(BoundingBox.Get_node(cube_app_IDX));
			uint64_t detail_idx = cube_app_IDX;

			multCounter = 1;
			multCounter_2 = 1;

			// Get the indeces at the bounding box:
			for (uint16_t d = 0; d < dimensions; d++) {
				uint64_t temp = floorf(positive_rem(k, multCounter * 2) / multCounter) * rescaling / 2;	// j-th component index

				detail_idx += temp * multCounter_2;
				multCounter *= 2;
				multCounter_2 *= total_signal_nodes;
			}

			assigned_node_indeces[detail_idx] = detail_idx;

			if (abs(signal[detail_idx]) >= tolerance) {
				assigned_node_markers[detail_idx] = 1;
			}
		}
	};
};


template<typename floatType>
class write_signal_in_bounding_box {
public:
	const floatType* input_signal;
	floatType* output_signal;
	const Mesh signal_domain;
	const Mesh signal_bounding_box;
	const uint64_t max_nodes;
public:
	deviceFunction void operator()(const uint64_t global_id) const {
		if (global_id >= max_nodes) { return; }

		// First, find the node index at the problem domain:
		uint64_t node_index_at_domain = signal_domain.Get_binIdx(signal_bounding_box.Get_node(global_id));

		// Now, fill the value of the output_signal with the value of the input_signal
		output_signal[global_id] = input_signal[node_index_at_domain];
	};
};


template<typename floatType>
class waveletMeshRefinement {
public:
	// Default constructor and destrucor
	waveletMeshRefinement();
	~waveletMeshRefinement();

private:
	// Input signal for wavelet transform: 
	floatType* __initial_signal;
	floatType* __initial_signal_dvc;

	// Parameters
	uint16_t __min_refinement_level;
	uint16_t __max_refinement_level;

	// Output signals
	floatType* __transformed_signal;
	floatType* __transformed_signal_dvc;
	floatType* __threshold_cutoff_transformed_signal;
	floatType* __threshold_cutoff_transformed_signal_dvc;

	// Output AMR arrays for indeces and assigned nodes
	/*uint64_t* __assigned_node_indeces;
	uint64_t* __assigned_node_indeces_dvc;
	uint32_t* __assigned_node_markers;
	uint32_t* __assigned_node_markers_dvc;*/

	// Other signal definitions:
	uint16_t __signal_dimension;

public:
	hostFunction
		void set_signal_dimension(uint16_t input);

	hostFunction deviceFunction
		uint16_t signal_dimension() const;

	hostFunction
		void set_min_refinement_level(uint16_t input);

	hostFunction deviceFunction
		uint16_t min_refinement_level() const;

	hostFunction
		void set_max_refinement_level(uint16_t input);

	hostFunction deviceFunction
		uint16_t max_refinement_level() const;

	hostFunction
		void set_initial_signal_host2dvc(floatType* input_signal);

	hostFunction
		void set_initial_signal_dvc2dvc(floatType* input_signal_dvc);

	hostFunction deviceFunction
		uint32_t nodes_per_dimension() const;

	hostFunction deviceFunction
		uint64_t total_signal_nodes() const;

	hostFunction
		floatType* initial_signal() const;

	hostFunction
		floatType* initial_signal_dvc() const;

	hostFunction
		floatType* transformed_signal() const;

	hostFunction
		floatType* transformed_signal_dvc() const;

	hostFunction
		void compute_wavelet_transform();
};

hostFunction
int setInitialParticles(
	const floatType* input_signal_dvc,
	Particle* output_active_nodes_dvc,
	const Mesh& signal_bounding_box,
	const Mesh& signal_domain
);