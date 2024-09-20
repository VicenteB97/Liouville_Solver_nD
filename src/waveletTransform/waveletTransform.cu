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
	// Change all of these for shared pointers? Or unique pointers and we pass ownership?
	// Instantiate host pointers as null pointers (Or should I do raw_pointers?)
	std::unique_ptr<floatType> m_initialSignal = std::make_unique<floatType>();
	std::unique_ptr<floatType> m_transformedSignal = std::make_unique<floatType>();
	std::unique_ptr<floatType> m_thresholdCutoffTransformedSignal = std::make_unique<floatType>();
	std::unique_ptr<uint64_t> m_assignedNodeIndeces = std::make_unique<uint64_t>();
	std::unique_ptr<uintType> m_assignedNodeMarkers = std::make_unique<uintType>();

	// Instantiate device pointers as null pointers
	deviceUniquePtr<floatType> m_initialSignal_dvc;
	deviceUniquePtr<floatType> m_transformedSignal_dvc;
	deviceUniquePtr<floatType> m_thresholdCutoffTransformedSignal_dvc;
	deviceUniquePtr<uint64_t> m_assignedNodeIndeces_dvc;
	deviceUniquePtr<uintType> m_assignedNodeMarkers_dvc;
};

hostFunction
waveletTransform::~waveletTransform() {
	// Delete dynamically-allocated host pointers
	// No need to manually delete them because we're working with
	// smart pointers in host and our smart unique_pointer in device!
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
void waveletTransform::setMinRefinementLevel(uint16_t input) {
	m_minRefinementLevel = input;
};

hostFunction deviceFunction
uint16_t waveletTransform::min_refinement_level() const {
	return m_minRefinementLevel;
};

hostFunction
void waveletTransform::setMaxRefinementLevel(uint16_t input) {
	m_maxRefinementLevel = input;
};

hostFunction deviceFunction
uint16_t waveletTransform::max_refinement_level() const {
	return m_maxRefinementLevel;
};

hostFunction
/// @brief We want to initialize the device signal from a signal living in host memory
/// @return 0 if execution was error-free, 1 otherwise
uint16_t waveletTransform::set_initial_signal_host2dvc(const floatType* input_signal) {
	// malloc and memcopy
	uint64_t copy_size_bytes = sizeof(floatType) * this->total_signal_nodes();
	
	try {
		m_initialSignal_dvc.malloc(this->total_signal_nodes(), (floatType)0); // allocate memory
		gpu_device.memCpy_hst2dvc((void*)m_initialSignal_dvc.get(), (void*)input_signal, copy_size_bytes);
	}
	catch (const std::exception& except) {
		std::cerr << "Caught exception at wavelet transform: " << except.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
};

hostFunction
uint16_t waveletTransform::setInitialSignal_dvc2dvc(const floatType* inputSignal_dvc) {
	// malloc and memcopy
	uint64_t copy_size_bytes = sizeof(floatType) * this->total_signal_nodes();

	try {
		m_initialSignal_dvc.malloc(this->total_signal_nodes(), (floatType)0);
		gpu_device.memCpy_dvc2dvc((void*)m_initialSignal_dvc.get(), (void*)inputSignal_dvc, copy_size_bytes);
	}
	catch (const std::exception& except) {
		std::cerr << "Exception thrown: " << except.what() << std::endl;	// That way we make sure that no exceptions escape these methods
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
};

// Rethink all of this!
hostFunction
floatType* waveletTransform::initialSignal_ptr() const {
	return m_initialSignal.get();
};

hostFunction
floatType* waveletTransform::initialSignal_dvcPtr() const {
	return m_initialSignal_dvc.get();
};

hostFunction deviceFunction
uintType waveletTransform::nodes_per_dim() const {
	return pow(2, m_maxRefinementLevel);
};

hostFunction deviceFunction
uint64_t waveletTransform::total_signal_nodes() const {
	return pow(this->nodes_per_dim(), __signal_dimension);
};

hostFunction
floatType* waveletTransform::transformed_signal_dvc() const {
	return m_transformedSignal_dvc.get();
};

hostFunction
floatType* waveletTransform::transformed_signal() const {
	// Note that this will be read from the GPU!!

	uint64_t copy_size_bytes = this->total_signal_nodes() * sizeof(floatType);
	gpu_device.memCpy_dvc2hst(m_transformedSignal.get(), m_transformedSignal_dvc.get(), copy_size_bytes);
	return m_transformedSignal.get();
};

hostFunction
uint64_t* waveletTransform::assigned_node_indeces() const {
	uint64_t copy_size_bytes = this->total_signal_nodes() * sizeof(uint64_t);
	gpu_device.memCpy_dvc2hst(m_assignedNodeIndeces.get(), m_assignedNodeIndeces_dvc.get(), copy_size_bytes);
	return m_assignedNodeIndeces.get();
};

hostFunction
uintType* waveletTransform::assigned_node_markers() const {
	uint64_t copy_size_bytes = this->total_signal_nodes() * sizeof(uintType);
	gpu_device.memCpy_dvc2hst(m_assignedNodeMarkers.get(), m_assignedNodeMarkers_dvc.get(), copy_size_bytes);
	return m_assignedNodeMarkers.get();
};

hostFunction
uint64_t* waveletTransform::assigned_node_indeces_dvc() const {
	return m_assignedNodeIndeces_dvc.get();
};

hostFunction
uintType* waveletTransform::assigned_node_markers_dvc() const {
	return m_assignedNodeMarkers_dvc.get();
};

hostFunction
void waveletTransform::computeWaveletTransform() {
	// Here, you've got to compute the wavelet transformation of the initial signal.
	uintType rescaling{ 2 };
	const uint64_t total_signal_nodes{ this->total_signal_nodes() };
	const uintType nodes_per_dim{ this->nodes_per_dim() };
	const double tolerance{ TOLERANCE_AMR }; 

	// Allocate memory 
	m_assignedNodeIndeces_dvc.malloc(total_signal_nodes, 0);
	m_assignedNodeMarkers_dvc.malloc(total_signal_nodes, 0);
	m_transformedSignal_dvc.malloc(total_signal_nodes, (floatType)0);

	gpu_device.memCpy_dvc2dvc(
		m_transformedSignal_dvc.get(),
		m_initialSignal_dvc.get(),
		sizeof(floatType) * total_signal_nodes
	);

	for (uint16_t k = 0; k < m_maxRefinementLevel - m_minRefinementLevel + 1; ++k) {

		uint16_t Threads = fmin(THREADS_P_BLK, total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS));
		uint64_t Blocks = floor((total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS) - 1) / Threads) + 1;

		gpu_device.launchKernel(Blocks, Threads,
			single_block_single_level_wavelet_transform{
				m_transformedSignal_dvc.get(),
				rescaling,
				nodes_per_dim,
				total_signal_nodes
			}
		);

		gpu_device.launchKernel(Blocks, Threads,
			get_nodes_above_threshold{
				m_transformedSignal_dvc.get(),
				m_assignedNodeIndeces_dvc.get(),
				m_assignedNodeMarkers_dvc.get(),
				rescaling,
				nodes_per_dim,
				total_signal_nodes,
				tolerance
			}
		);
		rescaling *= 2;	// our cartesianMesh will now have half the number of points
	}
};


hostFunction
uintType waveletTransform::sorted_assigned_nodes() {
	uint64_t total_nr_of_nodes(this->total_signal_nodes());

	thrust::device_ptr<uintType> assignedNodeMarkers_auxPtr(m_assignedNodeMarkers_dvc.get());
	thrust::device_ptr<uintType> assignedNodeIndeces_auxPtr(m_assignedNodeIndeces_dvc.get());

	const uintType nr_selected_nodes(
		thrust::reduce(
			thrust::device,
			assignedNodeMarkers_auxPtr,
			assignedNodeMarkers_auxPtr + total_nr_of_nodes
		)
	);

	// Set the selected nodes first, so that we collect the first nr_selected_nodes mesh indeces
	thrust::sort_by_key(
		thrust::device,
		assignedNodeMarkers_auxPtr,
		assignedNodeMarkers_auxPtr + total_nr_of_nodes,
		assignedNodeIndeces_auxPtr,
		thrust::greater<uintType>()
	);

	return nr_selected_nodes;
};