#include "Adapt_Mesh.cuh"

gpuDevice device;

hostFunction
int32_t setInitialParticles(
	const floatType* input_signal_dvc,
	Particle* output_active_nodes_dvc,
	const cartesianMesh& signal_bounding_box,
	const cartesianMesh& signal_domain
) {

	// Create the signal in the bounding box. Initialized to 0
	const uint64_t size_input_signal = signal_domain.total_nodes();
	const uint64_t size_signal_in_bounding_box = signal_bounding_box.total_nodes();

	// Create and fill with 0 the signal_in_bounding_box array (remember to free memory afterwards):
	thrust::device_vector<floatType> signal_in_bounding_box_dvc(size_signal_in_bounding_box);

	std::cout << "Size of initial PDF after memory allocation: " << signal_in_bounding_box_dvc.size() << std::endl;

	//Fill the signal_in_bounding_box
	uint16_t threads = fmin(THREADS_P_BLK, size_signal_in_bounding_box);
	uint64_t blocks = floor((size_signal_in_bounding_box - 1) / threads) + 1;

	gpu_device.launch_kernel(blocks, threads, write_signal_in_bounding_box{
		   input_signal_dvc,
		   rpc(signal_in_bounding_box_dvc, 0),
		   signal_domain,
		   signal_bounding_box,
		   size_signal_in_bounding_box
		}
	);

	// Create amr_handle
	waveletTransform amr_handle;

	amr_handle.set_min_refinement_level(0);
	amr_handle.set_max_refinement_level(log2(signal_bounding_box.__nodes_per_dim));
	amr_handle.set_initial_signal(signal_in_bounding_box_dvc);

	amr_handle.compute_wavelet_transform();
	amr_handle.get_detail_above_threshold_nodes(output_active_nodes_dvc, signal_bounding_box);

	return amr_handle.transformed_signal();
};

hostFunction
void get_detail_above_threshold_nodes(Particle* particle_locations, const cartesianMesh& signal_domain) const {

	// Here we assume that the compute_wavelet transform function has already been called
	// Get the number of assigned nodes
	uint64_t total_nr_of_nodes = sizeof(__assigned_node_markers) / sizeof(uint32_t);
	std::cout << "Total nr of nodes " << sizeof(__assigned_node_markers) << ".\n";

	const uintType nrSelectedNodes = thrust::reduce(thrust::device, __assigned_node_markers, __assigned_node_markers + total_nr_of_nodes);

	// Set the selected nodes first
	thrust::sort_by_key(
		thrust::device,
		__assigned_node_markers,
		__assigned_node_markers + total_nr_of_nodes,
		__assigned_node_indeces,
		thrust::greater<intType>()
	);

	//device.resize<Particle>(particle_locations, nrSelectedNodes);

	const intType Threads = fmin(THREADS_P_BLK, nrSelectedNodes);
	const intType Blocks = floor((nrSelectedNodes - 1) / Threads) + 1;

	device.launch_kernel(Blocks, Threads,
		customAssignToGpuArray<ELEMENTS_AT_A_TIME>{
			particle_locations,
				signal_domain,
				__assigned_node_indeces,
				nrSelectedNodes
		}
	);
};