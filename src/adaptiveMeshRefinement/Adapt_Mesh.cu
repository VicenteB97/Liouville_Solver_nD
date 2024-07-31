#include "Adapt_Mesh.cuh"

gpuDevice device;

deviceFunction
void write_signal_in_bounding_box::operator()(const uint64_t global_id) const {
	// Write function
};


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
	floatType* signal_in_bounding_box_dvc = nullptr;
	try {
		gpu_device.device_malloc((void**)&signal_in_bounding_box_dvc, size_signal_in_bounding_box * sizeof(floatType));
		gpu_device.device_memSet(signal_in_bounding_box_dvc, 0, size_signal_in_bounding_box * sizeof(floatType));

		//Fill the signal_in_bounding_box_dvc
		uint16_t threads = fmin(THREADS_P_BLK, size_signal_in_bounding_box);
		uint64_t blocks = floor((size_signal_in_bounding_box - 1) / threads) + 1;

		gpu_device.launch_kernel(blocks, threads, write_signal_in_bounding_box{
			   input_signal_dvc,
			   signal_in_bounding_box_dvc,
			   signal_domain,
			   signal_bounding_box,
			   size_signal_in_bounding_box
			}
		);

		gpu_device.device_free(signal_in_bounding_box);
	}
	catch (const std::exception& e) {
		std::cout << "Caught exception: " << e.what() << std::endl;
		if (signal_in_bounding_box_dvc) {
			gpu_device.device_free(signal_in_bounding_box_dvc);
		}
		return EXIT_FAILURE;
	}

	// Create amr_handle
	waveletTransform amr_engine;

	amr_engine.set_min_refinement_level(0);
	amr_engine.set_max_refinement_level(log2(signal_bounding_box.nodes_per_dim()));
	amr_engine.set_initial_signal_dvc2dvc(signal_in_bounding_box_dvc);

	amr_engine.compute_wavelet_transform();
	get_detail_above_threshold_nodes(amr_engine, output_active_nodes_dvc, signal_bounding_box);

	//return amr_engine.transformed_signal();
	return EXIT_SUCCESS;
};

hostFunction
void get_detail_above_threshold_nodes(waveletTransform& amr_engine, Particle* particle_locations, const cartesianMesh& signal_domain) {

	// We get the number of selected nodes because we'll read the first nr_selected_nodes indeces in the bounding box mesh
	uint32_t nr_selected_nodes = amr_engine.sorted_assigned_nodes();


	const uint32_t Threads = fmin(THREADS_P_BLK, nr_selected_nodes);
	const uint64_t Blocks = floor((nr_selected_nodes - 1) / Threads) + 1;

	device.launch_kernel(Blocks, Threads,
		get_nodes_from_indeces<ELEMENTS_AT_A_TIME>{
			particle_locations,
			signal_domain,
			amr_engine.assigned_node_indeces_dvc(),
			nr_selected_nodes
		}
	);
};