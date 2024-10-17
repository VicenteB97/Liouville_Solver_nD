#include "Adapt_Mesh.hpp"

deviceFunction
void writeSignalInBoundingBox::operator()(const uint64_t global_id) const {
	if (global_id >= max_nodes) { return; }

	// First, find the node index at the problem domain:
	uint64_t node_index_at_domain = signalDomain.getBinIdx(signalBoundingBox.get_node(global_id));

	// Now, fill the value of the output_signal with the value of the input_signal
	output_signal[global_id] = input_signal[node_index_at_domain];
};


hostFunction
void setInitialParticles(
	const floatType* inputSignal_dvc,
	deviceUniquePtr<Particle>& outputActiveNodes_dvc,
	deviceUniquePtr<floatType>& outputActiveNodesValues_dvc,
	const cartesianMesh& signalBoundingBox,
	const cartesianMesh& signalDomain
) {
	// Create the signal in the bounding box. Initialized to 0
	const uint64_t nodesSignalInBoundingBox = signalBoundingBox.total_nodes();

	// Create and fill with 0 the signal_in_bounding_box array:
	deviceUniquePtr<floatType> signalInBoundingBox_dvc(nodesSignalInBoundingBox, (floatType)0);

	//Fill the signalInBoundingBox_dvc
	uint16_t threads = fmin(THREADS_P_BLK, nodesSignalInBoundingBox);
	uint64_t blocks = ceil((double)(nodesSignalInBoundingBox - 1) / threads);

	try{
		gpu_device.launchKernel(blocks, threads,
			writeSignalInBoundingBox{
			   inputSignal_dvc,
			   signalInBoundingBox_dvc.get(),	// We will overwrite the info on this pointer
			   signalDomain,
			   signalBoundingBox,
			   nodesSignalInBoundingBox
			}
		);
	}
	catch (const std::exception& except) {
		mainTerminal.print_message("Exception caught writing singal into bounding box: " + std::string{ except.what() });
		throw;
	}

	// Use the specific AMR engine required: in this case, wavelet transform
	waveletTransform amrEngine;

	amrEngine.setSignalDomain(signalBoundingBox);
	amrEngine.setInitialSignal_dvc2dvc(signalInBoundingBox_dvc.get());

	try{
		amrEngine.computeWaveletTransform();
	}
	catch (const std::exception& except) {
		mainTerminal.print_message("Error has occurred at compute Wavelet Transform");
		throw;
	}
	
	getDetailAboveThresholdNodes(
		amrEngine, 
		outputActiveNodes_dvc, 
		outputActiveNodesValues_dvc, 
		signalBoundingBox
	);

	return /*EXIT_SUCCESS*/;
};


hostFunction
void getDetailAboveThresholdNodes(
	waveletTransform& amrEngine, 
	deviceUniquePtr<Particle>& particleLocations_dvc, 
	deviceUniquePtr<floatType>& outputActiveNodesValues_dvc,
	const cartesianMesh& signalDomain
) {

	// We get the number of selected nodes because we'll read the first nr_selected_nodes indeces in the bounding box mesh
	uintType nr_selected_nodes = amrEngine.sorted_assigned_nodes();
	particleLocations_dvc.malloc(nr_selected_nodes, Particle());
	outputActiveNodesValues_dvc.malloc(nr_selected_nodes, (floatType)0.0);

	try {
		const uint16_t Threads = fmin(THREADS_P_BLK, nr_selected_nodes / ELEMENTS_AT_A_TIME);
		if (Threads == 0) { throw std::invalid_argument("0 threads assigned at getDetailAboveThresholdNodes.\n"); }
		const uint64_t Blocks = floor((nr_selected_nodes / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;

		gpu_device.launchKernel(Blocks, Threads,
			get_nodes_from_indeces<ELEMENTS_AT_A_TIME>{
				particleLocations_dvc.get(),
				outputActiveNodesValues_dvc.get(),
				amrEngine.initialSignal_dvcPtr(),
				signalDomain,
				amrEngine.assigned_node_indeces_dvc(),
				nr_selected_nodes
			}
		);
	}
	catch (std::exception& e) {
		std::cout << "Caught exception: " << e.what() << std::endl;
		throw;
	}





	// FOR DEBUGGING:
	float* debugPtr = new float[nr_selected_nodes];

	gpu_device.memCpy_dvc2hst(debugPtr, outputActiveNodesValues_dvc.get(), sizeof(float) * nr_selected_nodes);
	std::unique_ptr<float[]> debugUPtr(debugPtr);

	std::ofstream file_0;
	file_0.open("Debug.csv");

	if (file_0.is_open()) {
		/*for (uint32_t k = 0; k < amrEngine.nodes_per_dim(); k++) {
			for (uint32_t j = 0; j < amrEngine.nodes_per_dim(); j++) {
				file_0 << debugPtr[j + amrEngine.nodes_per_dim() * k] << ",";
			}
			file_0 << "\n";
		}*/
		for (uint32_t k = 0; k < nr_selected_nodes; k++) {
			file_0 << debugUPtr[k] << ",";
		}
		file_0.close();
		mainTerminal.print_message("Completed");
	}
	else {
		mainTerminal.print_message("Failed");
	}
};