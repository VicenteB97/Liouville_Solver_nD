#include "waveletTransform/waveletTransform.hpp"

deviceFunction 
inline void haar_wavelet(floatType& s1, floatType& s2) {
	floatType aux = 0.5 * (s1 + s2);
	s2 = s1 - s2;
	s1 = aux;
};

deviceFunction 
void single_block_single_level_wavelet_transform::operator()(const uint64_t global_id) const {
	// Total nodes in each simple wavelet transform (per GPU thread)
	const uint16_t	miniSquareNodes = pow(2, dimensions);

	// Global index of the main approximation vertex at the bounding box
	int64_t cube_app_IDX = 0;

	// Compute the lowest corner node index
	uint64_t multCounter = 1;	// auxiliary counter: => pow(BoundingBox.__nodes_per_dim / in_rescaling, d)
	uint64_t multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.__nodes_per_dim, d)
	for (uint16_t d = 0; d < dimensions; d++) {
		int64_t temp_idx = floorf(positive_rem(global_id, multCounter * (total_signal_nodes / in_rescaling)) / multCounter) * in_rescaling;

		cube_app_IDX += temp_idx * multCounter_2;
		multCounter *= total_signal_nodes / in_rescaling;
		multCounter_2 *= total_signal_nodes;
	}

	multCounter = 1;	// Reinitialize for next computations: => pow(2, d)
	multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.__nodes_per_dim, d)

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
			uint64_t multCounter_4 = 1;	// => pow(BoundingBox.__nodes_per_dim, j)

			for (uint16_t j = 0; j < dimensions; j++) {
				int64_t temp = floorf(positive_rem(k, multCounter_3 * 2) / multCounter_3) * in_rescaling / 2;	// j-th component index

				app_IDX_at_BBox += temp * multCounter_4;
				multCounter_3 *= 2;
				multCounter_4 *= total_signal_nodes;
			}
			// Compute corresponding detail node
			int64_t det_IDX_at_BBox = app_IDX_at_BBox + multCounter_2 * in_rescaling / 2;
			// Compute the wavelet transform in-place
			haar_wavelet(signal[app_IDX_at_BBox], signal[det_IDX_at_BBox]);
		}
		multCounter *= 2;
		multCounter_2 *= in_nodes_per_dim;
	}
}

deviceFunction 
void get_nodes_above_threshold::operator()(const uint64_t global_id) const {
	if (global_id >= total_signal_nodes / powf(rescaling, dimensions)) {
		return;
	}
	// Total nodes in the problem domain
	const uint16_t	miniSquareNodes = pow(2, dimensions);

	// Global index of the main approximation vertex at the bounding box
	int64_t cube_app_IDX = 0;

	uint64_t multCounter = 1;	// auxiliary counter: => pow(BoundingBox.__nodes_per_dim / rescaling, d)
	uint64_t multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.__nodes_per_dim, d)
	for (uint16_t d = 0; d < dimensions; d++) {
		int64_t temp_idx = floorf(
			positive_rem(global_id, multCounter * (nodes_per_dim / rescaling)) / multCounter
		) * rescaling;

		cube_app_IDX += temp_idx * multCounter_2;
		multCounter *= nodes_per_dim / rescaling;
		multCounter_2 *= nodes_per_dim;
	}

	// Now we have to go see what happens with the outputs
	assigned_node_markers[cube_app_IDX] = 0;

	for (uint64_t k = 1; k < miniSquareNodes; k++) {
		// Particle visit_node(BoundingBox.get_node(cube_app_IDX));
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

hostFunction
waveletTransform::waveletTransform() {
	// Instantiate host pointers as null pointers
	__initial_signal = nullptr;
	__transformed_signal = nullptr;
	__threshold_cutoff_transformed_signal = nullptr;
	__assigned_node_indeces = nullptr;
	__assigned_node_markers = nullptr;

	// Instantiate device pointers as null pointers
	__initial_signal_dvc = nullptr;
	__transformed_signal_dvc = nullptr;
	__threshold_cutoff_transformed_signal_dvc = nullptr;
	__assigned_node_indeces_dvc = nullptr;
	__assigned_node_markers_dvc = nullptr;

};

hostFunction
waveletTransform::~waveletTransform() {
	// Delete dynamically-allocated host pointers
	if (__initial_signal != nullptr) {
		delete[] __initial_signal;
		__initial_signal == nullptr;
	};
	if (__transformed_signal != nullptr) {
		delete[] __transformed_signal;
		__transformed_signal == nullptr;
	};
	if (__threshold_cutoff_transformed_signal != nullptr) {
		delete[] __threshold_cutoff_transformed_signal;
		__threshold_cutoff_transformed_signal == nullptr;
	};
	if (__assigned_node_indeces != nullptr) {
		delete[] __assigned_node_indeces;
		__assigned_node_indeces == nullptr;
	};
	if (__assigned_node_markers != nullptr) {
		delete[] __assigned_node_markers;
		__assigned_node_markers == nullptr;
	};

	// Delete device pointers
	gpu_device.device_free(__initial_signal_dvc);
	gpu_device.device_free(__transformed_signal_dvc);
	gpu_device.device_free(__threshold_cutoff_transformed_signal_dvc);
	gpu_device.device_free(__assigned_node_indeces_dvc);
	gpu_device.device_free(__assigned_node_markers_dvc);

	// Destroy object
	delete this; 
};

hostFunction
void waveletTransform::set_signal_dimension(uint16_t input) {
	__signal_dimension = input;
};

hostFunction deviceFunction
uint16_t waveletTransform::signal_dimension() const {
	return __signal_dimension;
};

hostFunction
void waveletTransform::set_min_refinement_level(uint16_t input) {
	__min_refinement_level = input;
};

hostFunction deviceFunction
uint16_t waveletTransform::min_refinement_level() const {
	return __min_refinement_level;
};

hostFunction
void waveletTransform::set_max_refinement_level(uint16_t input) {
	__max_refinement_level = input;
};

hostFunction deviceFunction
uint16_t waveletTransform::max_refinement_level() const {
	return __max_refinement_level;
};

hostFunction
void waveletTransform::set_initial_signal_host2dvc(const floatType* input_signal) {
	// malloc and memcopy
	uint64_t copy_size_bytes = sizeof(floatType) * this->total_signal_nodes();
	gpu_device.device_malloc((void**)&__initial_signal_dvc, copy_size_bytes);
	gpu_device.memCpy_host_to_device(__initial_signal_dvc, (void*) input_signal, copy_size_bytes);
};

hostFunction
void waveletTransform::set_initial_signal_dvc2dvc(const floatType* input_signal_dvc) {
	// malloc and memcopy
	uint64_t copy_size_bytes = sizeof(floatType) * this->total_signal_nodes();
	gpu_device.device_malloc((void**)&__initial_signal_dvc, copy_size_bytes);
	gpu_device.memCpy_device_to_device(__initial_signal_dvc, (void*)input_signal_dvc, copy_size_bytes);
};

hostFunction
floatType* waveletTransform::initial_signal() const {
	return __initial_signal;
};

hostFunction
floatType* waveletTransform::initial_signal_dvc() const {
	return __initial_signal_dvc;
};

hostFunction deviceFunction
uint32_t waveletTransform::nodes_per_dim() const {
	return pow(2, __max_refinement_level);
};

hostFunction deviceFunction
uint64_t waveletTransform::total_signal_nodes() const {
	return pow(this->nodes_per_dim(), __signal_dimension);
};

hostFunction
floatType* waveletTransform::transformed_signal_dvc() const {
	return __transformed_signal_dvc;
};

hostFunction
floatType* waveletTransform::transformed_signal() const {
	uint64_t copy_size_bytes = this->total_signal_nodes() * sizeof(floatType);
	gpu_device.memCpy_device_to_host(__transformed_signal, __transformed_signal_dvc, copy_size_bytes);
	return __transformed_signal;
};

hostFunction
uint64_t* waveletTransform::assigned_node_indeces() const {
	uint64_t copy_size_bytes = this->total_signal_nodes() * sizeof(uint64_t);
	gpu_device.memCpy_device_to_host(__assigned_node_indeces, __assigned_node_indeces_dvc, copy_size_bytes);
	return __assigned_node_indeces;
};

hostFunction
uint32_t* waveletTransform::assigned_node_markers() const {
	uint64_t copy_size_bytes = this->total_signal_nodes() * sizeof(uint32_t);
	gpu_device.memCpy_device_to_host(__assigned_node_markers, __assigned_node_markers_dvc, copy_size_bytes);
	return __assigned_node_markers;
};

hostFunction
uint64_t* waveletTransform::assigned_node_indeces_dvc() const {
	return __assigned_node_indeces_dvc;
};

hostFunction
uint32_t* waveletTransform::assigned_node_markers_dvc() const {
	return __assigned_node_markers_dvc;
};

hostFunction
void waveletTransform::compute_wavelet_transform() {
	// Here, you've got to compute the wavelet transformation of the initial signal.
	// Pass the wavelet transform as a function pointer (future)
	uint32_t rescaling{ 2 };
	const uint64_t total_signal_nodes{ this->total_signal_nodes() };
	const uint32_t nodes_per_dim{ this->nodes_per_dim() };
	const double tolerance{ TOLERANCE_AMR }; 
	
	gpu_device.memCpy_device_to_device(
		__transformed_signal_dvc,
		__initial_signal_dvc,
		sizeof(floatType) * total_signal_nodes
	);

	// Allocate memory for the assigned node stuff
	gpu_device.device_malloc((void**)&__assigned_node_indeces_dvc, sizeof(uint64_t) * total_signal_nodes);
	gpu_device.device_malloc((void**)&__assigned_node_markers_dvc, sizeof(uint32_t) * total_signal_nodes);

	for (uint16_t k = 0; k < __max_refinement_level - __min_refinement_level + 1; ++k) {

		uint16_t Threads = fmin(THREADS_P_BLK, total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS));
		uint64_t Blocks = floor((total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS) - 1) / Threads) + 1;

		gpu_device.launch_kernel(Blocks, Threads,
			single_block_single_level_wavelet_transform{
				__transformed_signal_dvc,
				rescaling,
				nodes_per_dim,
				total_signal_nodes
			}
		);

		gpu_device.launch_kernel(Blocks, Threads,
			get_nodes_above_threshold{
				__transformed_signal_dvc,
				__assigned_node_indeces_dvc,
				__assigned_node_markers_dvc,
				rescaling,
				nodes_per_dim,
				total_signal_nodes,
				tolerance
			}
		);
		rescaling *= 2;	// our cartesianMesh will now have half the number of points
	}
};