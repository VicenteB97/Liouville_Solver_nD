//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//														//
//		Written by: Vicente José Bevia Escrig			//
//		Mathematics Ph.D. student (2020-2024) at:		//
//		Instituto de Matemática Multidisciplinar,		//
//		Universitat Politècnica de València, Spain		//
//														//
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

#include "ivpSolver.hpp"

using namespace thrust::placeholders;

// Set the default constructor. Parametric constructor won't be required!!
ivpSolver::ivpSolver() {};
ivpSolver::~ivpSolver() {};

// This method is used to build the domain in which the simulation will take place
int16_t ivpSolver::buildDomain() {

	// This variable represents the problem domain, which is NOT going to be the one used for computations
	__problem_domain.set_nodes_per_dimension(pow(2, FINEST_DISCR_LVL));

	return 0;
};

// Use the time information 
int16_t ivpSolver::buildTimeVec() {

	errorCheck(BuildTimeVector(__reinitialization_info, __delta_t, __reinitialization_steps))
	__storage_steps = SAVING_STEPS;

	const uintType savingArraySize = ceil((double) __reinitialization_info.size() / __storage_steps);

	__simulation_storage.resize(__problem_domain.total_nodes() * savingArraySize);

	return EXIT_SUCCESS;
};

// Build the Distribution information for all random parameters
int16_t ivpSolver::buildDistributions() {

	// Read initial probability distribution information from the Case_definition file
	// Inform about the case when truncation intervals are incorrectly chosen!
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		__initial_condition_distributions[d].Name = IC_NAMES[d];
		__initial_condition_distributions[d].isTruncated = IC_isTRUNC[d];
		__initial_condition_distributions[d].trunc_interval[0] = IC_InfTVAL[d];
		__initial_condition_distributions[d].trunc_interval[1] = IC_SupTVAL[d];
		__initial_condition_distributions[d].params[0] = IC_MEAN[d];
		__initial_condition_distributions[d].params[1] = IC_STD[d];
		__initial_condition_distributions[d].num_Samples = -1;		// This value is just for information. Samples are not chosen here
	}

	// Read parameter's probability distribution information from the Case_definition file
	for (uint16_t p = 0; p < PARAM_SPACE_DIMENSIONS; p++) {
		__parameter_distributions[p].Name = _DIST_NAMES[p];		// N, G or U distributions
		__parameter_distributions[p].isTruncated = _DIST_isTRUNC[p];		// TRUNCATED?
		__parameter_distributions[p].trunc_interval[0] = _DIST_InfTVAL[p];		// min of trunc. interval
		__parameter_distributions[p].trunc_interval[1] = _DIST_SupTVAL[p]; 	// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
		__parameter_distributions[p].params[0] = _DIST_MEAN[p];		// mean
		__parameter_distributions[p].params[1] = _DIST_STD[p];			// std
		__parameter_distributions[p].num_Samples = _DIST_N_SAMPLES[p];
	}

	return EXIT_SUCCESS;
};

// This function contains the most important function of them all: The full numerical method!
int16_t ivpSolver::evolvePDF(const cudaDeviceProp& D_Properties) {
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 	PARAMETERS
	intType total_sample_count = 1, sum_samples_count = 0;
	intType samples_per_parameter[PARAM_SPACE_DIMENSIONS];

	for (uintType i = 0; i < PARAM_SPACE_DIMENSIONS; i++) {
		uintType temp = __parameter_distributions[i].num_Samples;

		// Total samples from the parameter space discretization
		total_sample_count *= temp;

		// Length of the compressed parameter array
		sum_samples_count += temp;

		// Array storing the number of samples for each parameter
		samples_per_parameter[i] = temp;
	}

	// Full parameter array (won't change in size here, so unique_ptr will make the trick)
	std::unique_ptr<parameterPair[]> parameter_mesh = std::make_unique<parameterPair[]>(sum_samples_count);

	// Build the parameter mesh from previous info
	if (RANDOMIZE<PARAM_SPACE_DIMENSIONS>(parameter_mesh.get(), __parameter_distributions)) { return EXIT_FAILURE; }
	
	cudaUniquePtr<parameterPair> parameter_mesh_dvc;
	gpu_device.memCpy_hst2dvc(parameter_mesh_dvc.get(), parameter_mesh.get(), sum_samples_count * sizeof(parameterPair));

	cudaUniquePtr<intType> samples_per_parameter_dvc;
	gpu_device.memCpy_hst2dvc(samples_per_parameter_dvc.get(), &samples_per_parameter[0], PARAM_SPACE_DIMENSIONS * sizeof(intType));

	// auxiliary variable that will be used for ensemble mean computation
	floatType sum_sample_val = 0;
	for (uintType i = 0; i < total_sample_count; i++) {
		Param_vec<PARAM_SPACE_DIMENSIONS> temp = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i, parameter_mesh.get(), samples_per_parameter);
		sum_sample_val += temp.Joint_PDF;
	}

	// Output to the CLI
	std::cout << "Total number of random samples: " << total_sample_count << ".\n";

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PROBLEM DOMAIN AND INITIAL PDF

	// This cartesianMesh will be defined by the support bounding box of the data.
	__particle_bounding_box.set_boundary_inf((Particle)IC_InfTVAL);
	__particle_bounding_box.set_boundary_sup((Particle)IC_SupTVAL);

	// Make it square for AMR-purposes
	__particle_bounding_box.Squarify();

	// Update the number of pts per dimension
	__particle_bounding_box.set_nodes_per_dimension(round((IC_SupTVAL[0] - IC_InfTVAL[0]) / __problem_domain.discr_length()));

	// PDF values at the fixed, high-res cartesianMesh (CPU)
	std::unique_ptr<floatType[]> pdf_values_at_problem_domain = std::make_unique<floatType[]>(__problem_domain.total_nodes()); /*new floatType(__problem_domain.total_nodes());*/

	// initialize the cartesianMesh and the PDF at the cartesianMesh nodes (change so as to change the parameters as well)
	errorCheck(PDF_INITIAL_CONDITION(__problem_domain, pdf_values_at_problem_domain.get(), __initial_condition_distributions));

	// Pass the pdf_values_at_problem_domain to the gpu:
	cudaUniquePtr<floatType> pdf_values_at_problem_domain_dvc(__problem_domain.total_nodes(), (floatType)0);
	try {
		gpu_device.memCpy_hst2dvc(
			pdf_values_at_problem_domain_dvc.get(),
			pdf_values_at_problem_domain.get(),
			__problem_domain.total_nodes() * sizeof(floatType)
		);
	}
	catch (std::exception& e) {
		std::cout << "Caught exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Now we make a slightly larger domain for the computations:
	cartesianMesh Expanded_Domain;

	// Expand 40 nodes appears to be just fine (MAKE IT BETTER)
	const uint16_t expansion_nodes = 40;
	Expanded_Domain.Expand_From(__problem_domain, expansion_nodes);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES DEFINITION

	const uintType nrNodesPerFrame = __problem_domain.total_nodes();

	// Max memory to be used (in bytes). 95% just in case
	const uintType MAX_BYTES_USEABLE = 0.95 * (D_Properties.totalGlobalMem - nrNodesPerFrame * sizeof(floatType));

	// The following consts don't need an explanation
	const uintType	ConjGrad_MaxSteps = 1000;
	const floatType	RBF_SupportRadius = DISC_RADIUS * __problem_domain.discr_length();

	// Full array storing appended particles for all parameter samples. Use std::vector because it changes quite often!
	std::vector<Particle>	Full_Particle_Locations;
	cudaUniquePtr<Particle>	Full_Particle_Locations_dvc;

	std::vector<floatType>	Full_Particle_Values;
	cudaUniquePtr<floatType> Full_Particle_Values_dvc;

	indicators::ProgressBar statusBar{ 
		indicators::option::BarWidth{35},
		indicators::option::ForegroundColor{indicators::Color::yellow},
		indicators::option::ShowElapsedTime{true},
		indicators::option::ShowRemainingTime{false},
		indicators::option::PrefixText{"[INFO] Running..."},
		indicators::option::Start{"["},
		indicators::option::Fill{"*"},
		indicators::option::Lead{"*"},
		indicators::option::Remainder{"-"},
		indicators::option::End{"]"}, 
	};

	// Resize the simulation logger
	__simulation_log.resize(__reinitialization_info.size() - 1);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// START OF SIMULATION STEPS AS SUCH

	// Simulation steps
	uintType iteration_count = 0;
	// Saved frames
	std::atomic<uintType> saved_frames = 0;	// We need atomic type because we'll be saving in a separate thread concurrently!

	// Aux variable to switch between step functions (Heaviside forcing) 
	uintType mode = 0;

	// IF THERE ARE DELTA TERMS
	#if IMPULSE_TYPE == 1
	uintType jumpCount = 0;	// auxiliary variable to know how many delta jumpCounts have passed
	#endif

	// IF THERE ARE HEAVISIDE TERMS WITH EXTRA PARAMETERS
	#if INCLUDE_XTRA_PARAMS
	thrust::device_vector<double>	Extra_Parameter(XTRA_PARAM_LENGTH);
	thrust::copy(&XTRA_PARAM[0], &XTRA_PARAM[XTRA_PARAM_LENGTH], Extra_Parameter.begin());
	#else
	thrust::device_vector<double> Extra_Parameter(0);
	#endif

	// Define concurrent saving lambda function
	auto concurrentSaving = [&saved_frames](const floatType* vSrc, std::vector<floatType>& vDst, const uintType iteration_count, const uintType __storage_steps, const uintType nrNodesPerFrame) {
		if ((iteration_count == 0 || iteration_count % __storage_steps == 0) && saved_frames < vDst.size() / nrNodesPerFrame - 1) {
			// Store info in cumulative variable
			std::copy(vSrc, vSrc + saved_frames * nrNodesPerFrame, vDst.begin());
			saved_frames++;
		}
	};

	// IN THIS LINE WE START WITH THE ACTUAL ITERATIONS OF THE LIOUVILLE EQUATION
	while (iteration_count < __reinitialization_info.size() - 1) {

		// Save previous frame concurrently, so no time is used for this
		std::thread storeFrame_worker(concurrentSaving, pdf_values_at_problem_domain.get(), std::ref(__simulation_storage), iteration_count, __storage_steps, nrNodesPerFrame);

		__simulation_log.LogFrames[iteration_count].simIteration = iteration_count;
		__simulation_log.LogFrames[iteration_count].simTime = __reinitialization_info[iteration_count].time;

		// select the first and last time value of the current iteration
		double t0 = __reinitialization_info[iteration_count].time; double tF = __reinitialization_info[iteration_count + 1].time;

		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// -------------------------- ADAPT. MESH REFINEMENT --------------------------------- //
		/////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////
		auto startTimeSeconds = std::chrono::high_resolution_clock::now();

		try{
			setInitialParticles(
				pdf_values_at_problem_domain_dvc.get(),
				Full_Particle_Locations_dvc,
				__particle_bounding_box,
				__problem_domain
			);
		}
		catch (const std::exception& except) {
			std::cerr << "Exception caught at setInitialParticles: " << except.what() << std::endl;

			storeFrame_worker.join();	// Make sure that the dispatched thread is not working when we return from the function
			return EXIT_FAILURE;
		}

		auto endTimeSeconds = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> durationSeconds = endTimeSeconds - startTimeSeconds;

		// Number of particles to advect
		uintType AMR_ActiveNodeCount = Full_Particle_Locations_dvc.size_count();

//		// To the Log file
//		__simulation_log.LogFrames[iteration_count].log_AMR_Time = durationSeconds.count();
//		__simulation_log.LogFrames[iteration_count].log_AMR_RelevantParticles = AMR_ActiveNodeCount;
//
//		#if ERASE_dPDF
//		// Clear the GPU-stored PDF for better memory availability
//		D_PDF_ProbDomain.clear();
//		#endif
//
//		/////////////////////////////////////////////////////////////////////////////////////////
//		/////////////////////////////////////////////////////////////////////////////////////////
//		// -------------------------- PT. SEARCH --------------------------------------------- //
//		/////////////////////////////////////////////////////////////////////////////////////////
//		/////////////////////////////////////////////////////////////////////////////////////////
//
//		// Maximum neighbors to search. Diameter number of points powered to the dimension
//		uintType MaxNeighborNum = round(fmin(pow(2 * round(DISC_RADIUS) + 1, PHASE_SPACE_DIMENSIONS), AMR_ActiveNodeCount));
//
//		// Compressed COO-style indexing of the sparse interpolation matrix
//		D_Mat_Indx.resize(MaxNeighborNum * AMR_ActiveNodeCount);
//		thrust::fill(D_Mat_Indx.begin(), D_Mat_Indx.end(), -1);
//		// Sparse interpolation matrix values
//		D_Mat_Vals.resize(MaxNeighborNum * AMR_ActiveNodeCount);
//		thrust::fill(D_Mat_Vals.begin(), D_Mat_Vals.end(), 0);
//
//
//		startTimeSeconds = std::chrono::high_resolution_clock::now();
//
//		errorCheck(particleNeighborSearch(D_Particle_Locations,
//			D_Particle_Values,
//			D_Mat_Indx,
//			D_Mat_Vals,
//			AMR_ActiveNodeCount,
//			MaxNeighborNum,
//			__particle_bounding_box,
//			RBF_SupportRadius));
//
//		endTimeSeconds = std::chrono::high_resolution_clock::now();
//		durationSeconds = endTimeSeconds - startTimeSeconds;
//
//		/////////////////////////////////////////////////////////////////////////////////////////
//		/////////////////////////////////////////////////////////////////////////////////////////
//		// -------------------------- INTERPOLATION ------------------------------------------ //
//		/////////////////////////////////////////////////////////////////////////////////////////
//		/////////////////////////////////////////////////////////////////////////////////////////
//		// Declare the solution of the interpolation vector (weights of the RBF functions)
//		D_lambdas.resize(AMR_ActiveNodeCount);
//		thrust::fill(D_lambdas.begin(), D_lambdas.end(), 0);
//
//		interpVectors.resize(AMR_ActiveNodeCount);
//
//		startTimeSeconds = std::chrono::high_resolution_clock::now();
//		intType iterations = CONJUGATE_GRADIENT_SOLVE(D_lambdas,
//			D_Mat_Indx,
//			D_Mat_Vals,
//			D_Particle_Values,
//			interpVectors,
//			AMR_ActiveNodeCount,
//			MaxNeighborNum,
//			ConjGrad_MaxSteps,
//			TOLERANCE_ConjGrad);
//		if (iterations == -1) { std::cout << "Convergence failure.\n"; break; }
//		endTimeSeconds = std::chrono::high_resolution_clock::now();
//		durationSeconds = endTimeSeconds - startTimeSeconds;
//
//
//		// To the Log file
//		__simulation_log.LogFrames[iteration_count].log_Interpolation_Time = durationSeconds.count();
//		__simulation_log.LogFrames[iteration_count].log_Interpolation_Iterations = iterations;
//
//		#if ERASE_auxVectors == true
//		// Clear the vectors to save memory
//		D_Mat_Indx.clear();
//		D_Mat_Vals.clear();
//		#endif
//
//		D_Particle_Values = D_lambdas;
//
//		if (__reinitialization_info[iteration_count].impulse) {
//			/////////////////////////////////////////////////////////////////////////////////////////
//			/////////////////////////////////////////////////////////////////////////////////////////
//			// -------------------------- DELTA/HEAVISIDE IMPULSE TERMS -------------------------- //
//			/////////////////////////////////////////////////////////////////////////////////////////
//			/////////////////////////////////////////////////////////////////////////////////////////
//		#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-floatType IMPULSE!
//
//			startTimeSeconds = std::chrono::high_resolution_clock::now();
//
//			errorCheck(IMPULSE_TRANSFORM_PDF(D_PDF_ProbDomain,
//				D_Particle_Locations,
//				D_Particle_Values,
//				__reinitialization_info[iteration_count],
//				jumpCount,
//				__problem_domain,
//				Expanded_Domain,
//				__particle_bounding_box));
//
//			endTimeSeconds = std::chrono::high_resolution_clock::now();
//			durationSeconds = endTimeSeconds - startTimeSeconds;
//
//			// Enter the information into the log information
//			__simulation_log.LogFrames[iteration_count].log_Advection_Time = durationSeconds.count();
//			__simulation_log.LogFrames[iteration_count].log_Advection_TotalParticles = durationSeconds.count();
//
//			jumpCount++;
//
//			// Send back to CPU
//			PDF_ProbDomain = D_PDF_ProbDomain;
//
//
//		#elif(IMPULSE_TYPE == 2)
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//// ----------------------------- HEAVISIDE/STEP IMPULSE ------------------------------ //
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//			mode++;
//
//		#elif(IMPULSE_TYPE != 0)
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//// -------------------------- ERROR:  UNDEFINED IMPULSE floatType ------------------------- //
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//			std::cout << "Error in 'Dynamics.cuh'. You are choosing an unavailable option. Go back to 'Case_definition.cuh' and re-check options for IMPULSE_TYPE.\n";
//			error_check = -1;
//			break;
//
//		#endif
//			continue;
//		}
//		else {
//			/////////////////////////////////////////////////////////////////////////////////////////
//			/////////////////////////////////////////////////////////////////////////////////////////
//			// -------------------------- SMOOTH PARTICLE INTEGRATION ---------------------------- //
//			/////////////////////////////////////////////////////////////////////////////////////////
//			/////////////////////////////////////////////////////////////////////////////////////////
//
//			// Max. memory requirements for next step
//			const uintType Bytes_per_sample = AMR_ActiveNodeCount * (sizeof(floatType) * 2 + sizeof(Particle));
//
//			// Set number of random samples to work with at the same time
//			uintType Samples_PerBlk = fmin((uintType)total_sample_count, MAX_BYTES_USEABLE / Bytes_per_sample);
//
//			// Number of blocks to simulate
//			uintType total_simulation_blocks = ceil((double)total_sample_count / Samples_PerBlk);
//
//			// For correct reinitialization
//			D_fixedParticles = D_Particle_Locations;
//
//			for (uintType b = 0; b < total_simulation_blocks; b++) {
//
//				// Parameter sample offset init. and final to account for the block position
//				uintType Sample_idx_offset_init = b * Samples_PerBlk;
//				uintType Sample_idx_offset_final = fmin((b + 1) * Samples_PerBlk, total_sample_count);
//
//				// Actual number of samples in current block
//				Samples_PerBlk = Sample_idx_offset_final - Sample_idx_offset_init;
//
//				// Total AMR-activated nodes in the current block
//				uintType ActiveNodes_PerBlk = Samples_PerBlk * AMR_ActiveNodeCount;
//
//				D_Particle_Locations.resize(ActiveNodes_PerBlk);
//				D_Particle_Values.resize(ActiveNodes_PerBlk);
//
//				for (uintType k = 0; k < Samples_PerBlk; k++) {
//					thrust::copy(thrust::device, &D_fixedParticles[0], &D_fixedParticles[AMR_ActiveNodeCount],
//						&D_Particle_Locations[k * AMR_ActiveNodeCount]);
//
//					thrust::copy(thrust::device, &D_lambdas[0], &D_lambdas[AMR_ActiveNodeCount],
//						&D_Particle_Values[k * AMR_ActiveNodeCount]);
//				}
//
//				/////////////////////////////////////////////////////////////////////////////////////////
//				/////////////////////////////////////////////////////////////////////////////////////////
//				// -------------------------- POINT ADVECTION ---------------------------------------- //
//				/////////////////////////////////////////////////////////////////////////////////////////
//				/////////////////////////////////////////////////////////////////////////////////////////
//				uint16_t Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
//				uintType Blocks = floor((double)(ActiveNodes_PerBlk - 1) / Threads) + 1;
//
//				startTimeSeconds = std::chrono::high_resolution_clock::now();
//				ODE_INTEGRATE << <Blocks, Threads >> > (
//					rpc(D_Particle_Locations, 0),
//					rpc(D_Particle_Values, 0),
//					rpc(D_Parameter_cartesianMesh, Sample_idx_offset_init),
//					rpc(samples_per_parameter_dvc, 0),
//					t0,
//					__delta_t,
//					tF,
//					AMR_ActiveNodeCount,
//					Samples_PerBlk,
//					mode,
//					rpc(Extra_Parameter, 0),
//					__problem_domain);
//			if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}
//				endTimeSeconds = std::chrono::high_resolution_clock::now();
//				durationSeconds = endTimeSeconds - startTimeSeconds;
//
//				// To the Log file
//				__simulation_log.LogFrames[iteration_count].log_Advection_Time = durationSeconds.count();
//				__simulation_log.LogFrames[iteration_count].log_Advection_TotalParticles = ActiveNodes_PerBlk;
//
//				__particle_bounding_box.update_bounding_box(D_Particle_Locations);
//
//				// COMPUTE THE SOLUTION "PROJECTION" INTO THE L1 SUBSPACE. THIS WAY, REINITIALIZATION CONSERVES VOLUME (=1)
//				if (PHASE_SPACE_DIMENSIONS < 5) {
//					floatType temp = thrust::reduce(thrust::device, D_Particle_Values.begin(), D_Particle_Values.end());
//					thrust::transform(D_Particle_Values.begin(), D_Particle_Values.end(), D_Particle_Values.begin(), Samples_PerBlk / temp * _1);
//				}
//
//				/////////////////////////////////////////////////////////////////////////////////////////
//				/////////////////////////////////////////////////////////////////////////////////////////
//				// -------------------------- REINITIALIZATION --------------------------------------- //
//				/////////////////////////////////////////////////////////////////////////////////////////
//				/////////////////////////////////////////////////////////////////////////////////////////
//
//				#if ERASE_dPDF
//				D_PDF_ProbDomain.resize(nrNodesPerFrame, 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
//				#else
//				thrust::fill(thrust::device, D_PDF_ProbDomain.begin(), D_PDF_ProbDomain.end(), 0);
//				#endif
//
//				Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
//				Blocks = floor((double)(ActiveNodes_PerBlk - 1) / Threads) + 1;
//
//				startTimeSeconds = std::chrono::high_resolution_clock::now();
//				RESTART_GRID_FIND_GN << < Blocks, Threads >> > (rpc(D_Particle_Locations, 0),
//					rpc(D_PDF_ProbDomain, 0),
//					rpc(D_Particle_Values, 0),
//					rpc(D_Parameter_cartesianMesh, 0),
//					rpc(samples_per_parameter_dvc, 0),
//					RBF_SupportRadius,
//					AMR_ActiveNodeCount,
//					Samples_PerBlk,
//					Sample_idx_offset_init,
//					__problem_domain,
//					Expanded_Domain);
//			if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}
//				endTimeSeconds = std::chrono::high_resolution_clock::now();
//				durationSeconds = endTimeSeconds - startTimeSeconds;
//
//				// To the Log file
//				__simulation_log.LogFrames[iteration_count].log_Reinitialization_Time = durationSeconds.count();
//			}
//
//			/////////////////////////////////////////////////////////////////////////////////////////
//			/////////////////////////////////////////////////////////////////////////////////////////
//			// -------------------------- STORE PDF INTO OUTPUT ARRAY ---------------------------- //
//			/////////////////////////////////////////////////////////////////////////////////////////
//			/////////////////////////////////////////////////////////////////////////////////////////
//
//			// Correction of any possible negative PDF values
//			uintType Threads = fmin(THREADS_P_BLK, nrNodesPerFrame / ELEMENTS_AT_A_TIME);
//			uintType Blocks = floor((double)(nrNodesPerFrame / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;
//
//			CORRECTION << <Blocks, Threads >> > (rpc(D_PDF_ProbDomain, 0), nrNodesPerFrame);
//		if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}
//
//			// Divide by the sum of the values of the parameter mesh to obtain the weighted mean
//			thrust::transform(D_PDF_ProbDomain.begin(), D_PDF_ProbDomain.end(), D_PDF_ProbDomain.begin(), 1.0f / sum_sample_val * _1); // we use the thrust::placeholders here (@ the last input argument)
//
//			// Send back to CPU
//			PDF_ProbDomain = D_PDF_ProbDomain;
//
//		}

		// Upadte simulation step
		iteration_count++;

		storeFrame_worker.join();

		statusBar.set_option(indicators::option::PostfixText{ "Iterations: " + std::to_string(iteration_count) + "/" + std::to_string(__reinitialization_info.size() - 1) });
		statusBar.set_progress((float)iteration_count / (__reinitialization_info.size() - 1) * 100);
	}

	/*thrust::copy(PDF_ProbDomain.begin(), PDF_ProbDomain.end(), &__simulation_storage[saved_frames * nrNodesPerFrame]);*/

	std::cout << termcolor::bold << termcolor::green << "[INFO] Completed successfully!" << std::endl;
	std::cout << termcolor::reset;

	std::string log_filename{CASE};
	log_filename += "_log_file"; 
	__simulation_log.writeSimulationLog_toFile(log_filename);
	// Exit current function
	return 0;
};

int16_t ivpSolver::writeFramesToFile(const double& simulationDuration) {
	// We have added the capability of automatically detecting the number of 1 GB files where we can store the simulation output
	bool saving_active = true;		// see if saving is still active
	int16_t error_check = 0;

	const uintType nrNodesPerFrame = __problem_domain.total_nodes();

	const uint64_t MEM_2_STORE = __simulation_storage.size() * sizeof(float);

	uintType number_of_frames_needed = __simulation_storage.size() / nrNodesPerFrame;

	uint64_t max_frames_file = MAX_FILE_SIZE_B / nrNodesPerFrame / sizeof(float);
	uintType number_of_files_needed = floor((double)(number_of_frames_needed - 1) / max_frames_file) + 1;

	std::cout << "\nSimulation time: " << simulationDuration << " seconds. ";

	if (number_of_files_needed == 0) {
		std::cout << "There has been a problem. No memory written. Exiting simulation.\n";
		saving_active = false;
		error_check = -1;
	}

	while (saving_active) {
		#if TERMINAL_INPUT_ALLOWED
		char ans;
		std::cout << "Total memory of simulation: " << (double) MEM_2_STORE / 1024 / 1024 << " MB. \n";
		std::cout << number_of_files_needed << " files required for total storage. Total frames: " << number_of_frames_needed << ", with frames per file: " << max_frames_file << " \n";
		std::cout << "Write? (Y = Yes(total), N = No, P = Partial): ";
		std::cin >> ans;

		while ((ans != 'N') && (ans != 'n') && (ans != 'Y') && (ans != 'y') && (ans != 'P') && (ans != 'p')) {
			std::cout << "Incorrect option. Choose one of the following (NOT case sensitive: Y = Yes, N = No, P = Partial): ";
			std::cin >> ans;
		}


		if ((ans != 'N') && (ans != 'n')) {

			intType frames_init = 0, frames_end = number_of_files_needed - 1;
			bool condition = false;

			if ((ans == 'P') || (ans == 'p')) {
				while (!condition) {
					std::cout << "Initial frame (must be >= 0): ";
					std::cin >> frames_init;
					std::cout << "Final frame (must be < " << number_of_frames_needed << "): ";
					std::cin >> frames_end;

					if (frames_init < 0 || frames_end >= number_of_frames_needed || frames_init > frames_end) {

						if (frames_init == -1 || frames_end == -1) {
							std::cout << "Exiting simulation without saving simulation results...\n";
							return -1;
						}
						std::cout << "Check numbers, something's not right...\n";
					}
					else {
						condition = true;
						number_of_frames_needed = frames_end - frames_init + 1;
						number_of_files_needed = (uintType) floor((double)(number_of_frames_needed - 1) / max_frames_file) + 1;
					}
				}
			}
			else {
				frames_init = 0, frames_end = number_of_files_needed - 1;
			}

			#pragma omp parallel for
			for (int16_t k = 0; k < (int16_t)number_of_files_needed; k++) {

				uintType frames_in_file = fmin(max_frames_file, number_of_frames_needed - k * max_frames_file);

				std::string temp_str = std::to_string((uintType)k);

				// SIMULATION INFORMATION FILE
				std::string source_path{SRC_DIR};

				std::string relative_pth = source_path + "/output/" + CASE;
				relative_pth.append("_Simulation_info_");
				relative_pth.append(temp_str);
				relative_pth.append(".csv");

				std::ofstream file1(relative_pth, std::ios::out);
				assert(file1.is_open());

				file1 << nrNodesPerFrame << "," << __problem_domain.__nodes_per_dim << ",";

				for (uintType d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
					file1 << __problem_domain.__boundary_inf.dim[d] << "," << __problem_domain.__boundary_sup.dim[d] << ",";
				}
				for (uint16_t d = 0; d < PARAM_SPACE_DIMENSIONS; d++) {
					file1 << __parameter_distributions[d].num_Samples << ",";
				}
				file1 << simulationDuration << "\n";

				for (uintType i = k * max_frames_file + frames_init; i < k * max_frames_file + frames_in_file + frames_init - 1; i++) {
					file1 << __reinitialization_info[i * __storage_steps].time << ",";
				}
				file1 << __reinitialization_info[(k * max_frames_file + frames_in_file + frames_init - 1) * __storage_steps].time;

				file1.close();

				// SIMULATION OUTPUT
				relative_pth = source_path + "/output/" + CASE;
				relative_pth.append("_Mean_PDFs_");
				relative_pth.append(temp_str);
				relative_pth.append(".bin");

				std::ofstream myfile(relative_pth, std::ios::out | std::ios::binary);
				assert(myfile.is_open());

				myfile.write((char*)&__simulation_storage[(k * max_frames_file + frames_init) * nrNodesPerFrame], sizeof(floatType) * frames_in_file * nrNodesPerFrame);
				myfile.close();
				std::cout << "Simulation output file " << k << " completed!\n";

			}
		}
		saving_active = false;
		#else
		std::cout << "Total memory of simulation: " << (double) MEM_2_STORE / 1024 / 1024 << " MB. \n";
		std::cout << number_of_files_needed << " files required for total storage. Total frames: ";
		std::cout << number_of_frames_needed << ", with frames per file: " << max_frames_file << " \n";

		intType frames_init = 0, frames_end = number_of_files_needed - 1;
		bool condition = false;

		if ((SAVING_TYPE == "P") || (SAVING_TYPE == "p")) {
			while (!condition) {
				frames_init = FIRST_FRAME;
				frames_end = LAST_FRAME;

				if (frames_init < 0 || frames_end >= number_of_frames_needed || frames_init > frames_end) {

					if (frames_init == -1 || frames_end == -1) {
						std::cout << "Exiting simulation without saving simulation results...\n";
						return -1;
					}
					std::cout << "Check numbers, something's not right...\n";
				}
				else {
					condition = true;
					number_of_frames_needed = frames_end - frames_init + 1;
					number_of_files_needed = floor((double)(number_of_frames_needed - 1) / max_frames_file) + 1;
				}
			}
		}

		#pragma omp parallel for
		for (int16_t k = 0; k < (int16_t)number_of_files_needed; k++) {

			uintType frames_in_file = fmin(max_frames_file, number_of_frames_needed - k * max_frames_file);

			std::string temp_str = std::to_string((uintType)k);

			// SIMULATION INFORMATION FILE
			std::string source_path{SRC_DIR};

			std::string relative_pth = source_path + "/output/" + CASE;
			relative_pth.append("_Simulation_info_");
			relative_pth.append(temp_str);
			relative_pth.append(".csv");

			std::ofstream file1(relative_pth, std::ios::out);
			assert(file1.is_open());

			file1 << nrNodesPerFrame << "," << __problem_domain.nodes_per_dim() << ",";

			for (uintType d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
				file1 << __problem_domain.boundary_inf().dim[d] << "," << __problem_domain.boundary_sup().dim[d] << ",";
			}
			for (uint16_t d = 0; d < PARAM_SPACE_DIMENSIONS; d++) {
				file1 << __parameter_distributions[d].num_Samples << ",";
			}
			file1 << simulationDuration << "\n";

			for (uintType i = k * max_frames_file + frames_init; i < k * max_frames_file + frames_in_file + frames_init - 1; i++) {
				file1 << __reinitialization_info[i * __storage_steps].time << ",";
			}

			// Store the last element of the time vector, even if it is not proportional to the number of saving steps
			if (number_of_files_needed == 1){
				file1 << __reinitialization_info.back().time;
			}
			else{
				file1 << __reinitialization_info[(k * max_frames_file + frames_in_file + frames_init - 1) * __storage_steps].time;
			}

			file1.close();

			// SIMULATION OUTPUT
			relative_pth = source_path + "/output/" + CASE;
			relative_pth.append("_Mean_PDFs_");
			relative_pth.append(temp_str);
			relative_pth.append(".bin");

			std::ofstream myfile(relative_pth, std::ios::out | std::ios::binary);
			assert(myfile.is_open());

			myfile.write((char*)&__simulation_storage[(k * max_frames_file + frames_init) * nrNodesPerFrame], sizeof(floatType) * frames_in_file * nrNodesPerFrame);
			myfile.close();

			std::string temp_output_str = "|	Simulation output file " + std::to_string(k) + " completed!";
			Intro_square_filler(temp_output_str, windowLength, 1, 0);
		}
		saving_active = false;
		#endif
	}
	return error_check;
};
