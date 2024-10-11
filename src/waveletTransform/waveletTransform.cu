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
	if (global_id >= total_signal_nodes / powf(in_rescaling, dimensions)) { return; }

	const uint64_t nodesInLevelPerDim = boundingBox.nodes_per_dim() / in_rescaling;

	// Build the cube where the wavelet transform will be computed
	uint16_t tempMultCounter = 1, tempPowCounter = 1;
	uint64_t lowestNodeIdx = 0;
	for (uint16_t d = 0; d < dimensions; d++) {
		lowestNodeIdx += tempPowCounter * in_rescaling *	// terms to multiply for the global index
			floorf(positive_rem(global_id, nodesInLevelPerDim * tempMultCounter) / tempMultCounter); // index in rescaled domain

		tempMultCounter *= nodesInLevelPerDim;
		tempPowCounter *= boundingBox.nodes_per_dim();
	}
	Particle lowestNode{ boundingBox.get_node(lowestNodeIdx) };
	Particle highestNode{ boundingBox.get_node(boundingBox.getBinIdx(lowestNode, (intType)(in_rescaling >> 1))) };

	cartesianMesh miniCubeWaveletTransform(lowestNode, highestNode);

	// Compute wavelet transform
	tempMultCounter = 1;								// Reinitialize for next computations: => pow(2, d)
	const uint16_t miniSquareNodes(1 << dimensions);	// Note that pow(2^dimensions) = 1 << dimensions

	for (uint16_t d = 0; d < dimensions; d++) {
		for (uint16_t k = 0; k < miniSquareNodes; k++) {
			bool isApproxNode = (floorf(positive_rem(k, tempMultCounter << 1) / tempMultCounter) == 0);
			if (!isApproxNode) { continue; }

			uint64_t approxNodeIdx = boundingBox.idx_here_from_other_mesh(k, miniCubeWaveletTransform);
			uint64_t detailNodeIdx = boundingBox.idx_here_from_other_mesh(k + (1 << d), miniCubeWaveletTransform);

			haar_wavelet(signal[approxNodeIdx], signal[detailNodeIdx]);
		}
		tempMultCounter <<= 1;
	}

	// Retrieve important nodes
	assigned_node_indeces[lowestNodeIdx] = lowestNodeIdx;
	for (uint16_t k = 1; k < miniSquareNodes; k++) {
		uint64_t detailNodeIdx = boundingBox.idx_here_from_other_mesh(k, miniCubeWaveletTransform);

		assigned_node_indeces[detailNodeIdx] = detailNodeIdx;
		if (abs(signal[detailNodeIdx]) < tolerance) {
			assigned_node_markers[detailNodeIdx] = 0;
		}
	}
};


hostFunction
waveletTransform::waveletTransform() : 
	m_initialSignal(),
	m_transformedSignal(),
	m_thresholdCutoffTransformedSignal(),
	m_assignedNodeIndeces(),
	m_assignedNodeMarkers(),
	m_initialSignal_dvc(),
	m_transformedSignal_dvc(),
	m_thresholdCutoffTransformedSignal_dvc(),
	m_assignedNodeIndeces_dvc(),
	m_assignedNodeMarkers_dvc(),
	__signal_dimension(PHASE_SPACE_DIMENSIONS),
	m_minRefinementLevel(0),
	m_maxRefinementLevel(1),
	m_signalDomain()
{};

hostFunction
waveletTransform::~waveletTransform() {
	// Delete dynamically-allocated host pointers
	// No need to manually delete them because we're working with
	// smart pointers in host and our smart unique_pointer in device!
};

hostFunction
void waveletTransform::setSignalDomain(const cartesianMesh& signalDomain) {
	m_signalDomain = signalDomain;
	m_maxRefinementLevel = round(log2(m_signalDomain.nodes_per_dim()));
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
		mainTerminal.print_message("Caught exception at wavelet transform: " + std::string{ except.what() });
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
		mainTerminal.print_message("Caught exception initiating device signal: " + std::string{ except.what() });	// That way we make sure that no exceptions escape these methods
		throw;
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
	m_assignedNodeMarkers_dvc.malloc(total_signal_nodes, 1);	// We choose all of them at the beginning
	m_transformedSignal_dvc.malloc(total_signal_nodes, (floatType)0);
	
	try{
		gpu_device.memCpy_dvc2dvc(
			m_transformedSignal_dvc.get(),
			m_initialSignal_dvc.get(),
			sizeof(floatType) * total_signal_nodes
		);
	}
	catch (const std::exception& except) {
		mainTerminal.print_message("Copying went wrong: " + std::string{ except.what() });
		throw;
	}

	for (uint16_t k = 0; k < m_maxRefinementLevel - m_minRefinementLevel; ++k) {

		uint64_t nodes_current_iteration = total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS);

		uint16_t Threads = fmin(THREADS_P_BLK, nodes_current_iteration);
		uint64_t Blocks = floor((nodes_current_iteration - 1) / Threads) + 1;
		try{
			gpu_device.launchKernel(Blocks, Threads,
				single_block_single_level_wavelet_transform{
					m_transformedSignal_dvc.get(),
					m_assignedNodeIndeces_dvc.get(),
					m_assignedNodeMarkers_dvc.get(),
					rescaling,
					nodes_per_dim,
					total_signal_nodes,
					tolerance,
					PHASE_SPACE_DIMENSIONS,
					m_signalDomain
				}
			);
		}
		catch (const std::exception& except) {
			mainTerminal.print_message("Exception caught at single block wavelet transform, level: " + std::to_string(k));
			mainTerminal.print_message("Exception info: " + std::string{ except.what() });
			throw;
		}

		rescaling <<= 1;	// or *=2; our cartesianMesh will now have half the number of points
	}
};


hostFunction
uintType waveletTransform::sorted_assigned_nodes() {
	uint64_t total_nr_of_nodes(this->total_signal_nodes());

	thrust::device_ptr<uintType> assignedNodeMarkers_auxPtr(m_assignedNodeMarkers_dvc.get());
	thrust::device_ptr<uint64_t> assignedNodeIndeces_auxPtr(m_assignedNodeIndeces_dvc.get());

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