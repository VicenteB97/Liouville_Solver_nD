#include "waveletTransform.h"

/// @brief (DEVICE FUNCTION) Compute a 1D Haar wavelet transform
/// @param s1 
/// @param s2 
/// @return
template<typename floatType>
deviceFunction inline void haar_wavelet(floatType& s1, floatType& s2) {
	floatType aux = 0.5 * (s1 + s2);
	s2 = s1 - s2;
	s1 = aux;
}

hostFunction
int32_t setInitialParticles(
	const floatType* input_signal_dvc,
	Particle* output_active_nodes_dvc,
	const Mesh& signal_bounding_box,
	const Mesh& signal_domain
) {
	// deviceClass device;
	// Create the signal in the bounding box. Initialized to 0
	const uint64_t size_input_signal = signal_domain.Total_Nodes();
	const uint64_t size_signal_in_bounding_box = signal_bounding_box.Total_Nodes();

	// Create and fill with 0 the signal_in_bounding_box array (remember to free memory afterwards):
	thrust::device_vector<floatType> signal_in_bounding_box_dvc(size_signal_in_bounding_box);

	std::cout << "Size of initial PDF after memory allocation: " << signal_in_bounding_box_dvc.size() << std::endl;

	// Fill the signal_in_bounding_box
	// uint16_t threads = fmin(THREADS_P_BLK, size_signal_in_bounding_box);
	// uint64_t blocks = floor((size_signal_in_bounding_box - 1) / threads ) + 1;

	// device.LaunchKernel(blocks, threads,
	// 	write_signal_in_bounding_box{
	// 		input_signal_dvc,
	// 		rpc(signal_in_bounding_box_dvc, 0),
	// 		signal_domain,
	// 		signal_bounding_box,
	// 		size_signal_in_bounding_box
	// 	}
	// );

	// Create amr_handle
	// waveletMeshRefinement amr_handle;

	// amr_handle.set_min_refinement_level(0);
	// amr_handle.set_max_refinement_level(log2(signal_bounding_box.Nodes_per_Dim));
	// amr_handle.set_initial_signal(signal_in_bounding_box_dvc);

	// amr_handle.compute_wavelet_transform();
	// amr_handle.get_detail_above_threshold_nodes(output_active_nodes_dvc, signal_bounding_box);

	// return amr_handle.transformed_signal();
	return 0;
}

hostFunction
waveletMeshRefinement::waveletMeshRefinement() {};

hostFunction
waveletMeshRefinement::~waveletMeshRefinement() { delete this; };

hostFunction
void waveletMeshRefinement::set_signal_dimension(uint16_t input) {
	__signal_dimension = input;
};

hostFunction deviceFunction
uint16_t waveletMeshRefinement::signal_dimension() const {
	return __signal_dimension;
};

hostFunction
void waveletMeshRefinement::set_min_refinement_level(uint16_t input) {
	__min_refinement_level = input;
};

hostFunction deviceFunction
uint16_t waveletMeshRefinement::min_refinement_level() const {
	return __min_refinement_level;
};

hostFunction
void waveletMeshRefinement::set_max_refinement_level(uint16_t input) {
	__max_refinement_level = input;
};

hostFunction deviceFunction
uint16_t waveletMeshRefinement::max_refinement_level() const {
	return __max_refinement_level;
};

hostFunction
void waveletMeshRefinement::set_initial_signal_host2dvc(floatType* input_signal) {
	__initial_signal_dvc = input_signal;
	// malloc and memcopy
};

hostFunction
void waveletMeshRefinement::set_initial_signal_dvc2dvc(floatType* input_signal_dvc) {
	__initial_signal_dvc = input_signal;
	// malloc and memcopy
};

hostFunction
floatType* waveletMeshRefinement::initial_signal() const {
	return __initial_signal;
};

hostFunction
floatType* waveletMeshRefinement::initial_signal_dvc() const {
	return __initial_signal_dvc;
};

hostFunction deviceFunction
uint32_t waveletMeshRefinement::nodes_per_dimension() const {
	return pow(2, __max_refinement_level);
};

hostFunction deviceFunction
uint64_t waveletMeshRefinement::total_signal_nodes() const {
	return pow(this->nodes_per_dimension(), __signal_dimension);
};

hostFunction
floatType* waveletMeshRefinement::transformed_signal() const {
	return __transformed_signal;
};	// Include a copy from the thrust container to the initial type

//hostFunction
//uint64_t* waveletMeshRefinement::assigned_node_indeces() const {
//	return __assigned_node_indeces;
//};
//
//hostFunction
//uint32_t* waveletMeshRefinement::assigned_node_marker() const {
//	return __assigned_node_markers;
//};

hostFunction
void waveletMeshRefinement::compute_wavelet_transform() {
	// Here, you've got to compute the wavelet transformation of the initial signal.
	// Pass the wavelet transform as a function pointer (future)
	__transformed_signal = __initial_signal;

	deviceClass device;

	uint32_t rescaling{ 2 };
	const uint64_t total_signal_nodes{ this->total_signal_nodes() };
	const uint32_t nodes_per_dimension{ this->nodes_per_dimension() };
	const double tolerance{ 1E-5 };

	for (uint16_t k = 0; k < __max_refinement_level - __min_refinement_level + 1; k++) {

		uint16_t Threads = fmin(THREADS_P_BLK, total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS));
		uint64_t Blocks = floor((total_signal_nodes / pow(rescaling, PHASE_SPACE_DIMENSIONS) - 1) / Threads) + 1;

		device.LaunchKernel(Blocks, Threads,
			single_block_single_level_wavelet_transform{
				__transformed_signal,
				rescaling,
				nodes_per_dimension,
				total_signal_nodes
			}
		);

		// Resize the arrays!!
		device.resize<uint32_t>(__assigned_node_markers, total_signal_nodes);
		device.resize<uint64_t>(__assigned_node_indeces, total_signal_nodes);

		device.LaunchKernel(Blocks, Threads,
			get_nodes_above_threshold{
				__transformed_signal,
				__assigned_node_indeces,
				__assigned_node_markers,
				rescaling,
				nodes_per_dimension,
				total_signal_nodes,
				tolerance
			}
		);
		rescaling *= 2;	// our Mesh will now have half the number of points
	}
};

//hostFunction
//void waveletMeshRefinement::get_detail_above_threshold_nodes(
//	Particle* particle_locations,
//	const Mesh& signal_domain
//) const {
//
//	deviceClass device;
//
//	// Here we assume that the compute_wavelet transform function has already been called
//	// Get the number of assigned nodes
//	uint64_t total_nr_of_nodes = sizeof(__assigned_node_markers) / sizeof(uint32_t);
//	std::cout << "Total nr of nodes " << sizeof(__assigned_node_markers) << ".\n";
//
//	const uintType nrSelectedNodes = thrust::reduce(thrust::device, __assigned_node_markers, __assigned_node_markers + total_nr_of_nodes);
//
//	// Set the selected nodes first
//	thrust::sort_by_key(
//		thrust::device,
//		__assigned_node_markers,
//		__assigned_node_markers + total_nr_of_nodes,
//		__assigned_node_indeces,
//		thrust::greater<intType>()
//	);
//
//	device.resize<Particle>(particle_locations, nrSelectedNodes);
//
//	const intType Threads = fmin(THREADS_P_BLK, nrSelectedNodes);
//	const intType Blocks = floor((nrSelectedNodes - 1) / Threads) + 1;
//
//	device.LaunchKernel(Blocks, Threads,
//		customAssignToGpuArray<ELEMENTS_AT_A_TIME>{
//		particle_locations,
//			signal_domain,
//			__assigned_node_indeces,
//			nrSelectedNodes
//	}
//	);
//};