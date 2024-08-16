#include "Adapt_Mesh.cuh"


deviceFunction
void write_signal_in_bounding_box::operator()(const uint64_t global_id) const {
	if (global_id >= max_nodes) { return; }

	// First, find the node index at the problem domain:
	uint64_t node_index_at_domain = signal_domain.get_bin_idx(signal_bounding_box.get_node(global_id));

	// Now, fill the value of the output_signal with the value of the input_signal
	output_signal[global_id] = input_signal[node_index_at_domain];
};


hostFunction
void setInitialParticles(
	const floatType* input_signal_dvc,
	cudaUniquePtr<Particle>& output_active_nodes_dvc,
	const cartesianMesh& signal_bounding_box,
	const cartesianMesh& signal_domain
) {
	// Create the signal in the bounding box. Initialized to 0
	const uint64_t nodes_signal_in_bounding_box = signal_bounding_box.total_nodes();

	// Create and fill with 0 the signal_in_bounding_box array (remember to free memory afterwards):
	cudaUniquePtr<floatType> signal_in_bounding_box_dvc(nodes_signal_in_bounding_box, 0);

	//Fill the signal_in_bounding_box_dvc
	uint16_t threads = fmin(THREADS_P_BLK, nodes_signal_in_bounding_box);
	uint64_t blocks = floor((nodes_signal_in_bounding_box - 1) / threads) + 1;

	gpu_device.launchKernel(blocks, threads,
		write_signal_in_bounding_box{
		   input_signal_dvc,
		   signal_in_bounding_box_dvc.get(),	// We will overwrite the info on this pointer
		   signal_domain,
		   signal_bounding_box,
		   nodes_signal_in_bounding_box
		}
	);

	// Use the specific AMR engine required: in this case, wavelet transform
	waveletTransform amr_engine;

	amr_engine.set_min_refinement_level(0);
	amr_engine.set_max_refinement_level(log2(signal_bounding_box.nodes_per_dim()));
	amr_engine.set_initial_signal_dvc2dvc(signal_in_bounding_box_dvc.get());

	amr_engine.compute_wavelet_transform();
	get_detail_above_threshold_nodes(amr_engine, output_active_nodes_dvc, signal_bounding_box);

	return /*EXIT_SUCCESS*/;
};


hostFunction
void get_detail_above_threshold_nodes(waveletTransform& amr_engine, cudaUniquePtr<Particle>& particle_locations_dvc, const cartesianMesh& signal_domain) {

	// We get the number of selected nodes because we'll read the first nr_selected_nodes indeces in the bounding box mesh
	uintType nr_selected_nodes = amr_engine.sorted_assigned_nodes();
	particle_locations_dvc.malloc(nr_selected_nodes);

	try {
		const uintType Threads = fmin(THREADS_P_BLK, nr_selected_nodes);
		if (Threads == 0) { throw std::invalid_argument("0 threads assigned at get_detail_above_threshold_nodes.\n"); }
		const uint64_t Blocks = floor((nr_selected_nodes - 1) / Threads) + 1;

		gpu_device.launchKernel(Blocks, Threads,
			get_nodes_from_indeces<ELEMENTS_AT_A_TIME>{
				particle_locations_dvc.get(),
				signal_domain,
				amr_engine.assigned_node_indeces_dvc(),
				nr_selected_nodes
			}
		);
	}
	catch (std::exception& e) {
		std::cout << "Caught exception: " << e.what() << std::endl;
		return;
	}
};