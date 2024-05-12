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

#include <ivpSolver.cuh>

using namespace thrust::placeholders;

// Set the default constructor. Parametric constructor won't be needed!
ivpSolver::ivpSolver() {};
ivpSolver::~ivpSolver() {};

// This method is used to build the domain in which the simulation will take place
int16_t ivpSolver::buildDomain() {

	int16_t LvlFine;

	#if TERMINAL_INPUT_ALLOWED
	bool getAnswer = true;
	std::string inputTerminal;

	while (getAnswer) {
		std::cout << "Finest level in the domain?: ";
		std::cin >> inputTerminal;

		errorCheck(intCheck(getAnswer, inputTerminal, DOMAIN_ERR_MSG, 1, 0))
	}

	LvlFine = std::stoi(inputTerminal);
	#else
	LvlFine = FINEST_DISCR_LVL;
	#endif

	// This variable represents the problem domain, which is NOT going to be the one used for computations
	__problem_domain.Nodes_per_Dim = pow(2, LvlFine);

	return 0;
};

// Use the time information 
int16_t ivpSolver::buildTimeVec() {

	errorCheck(BuildTimeVector(__reinitialization_info, __delta_t, __reinitialization_steps))

	#if TERMINAL_INPUT_ALLOWED
	// Build saving array
	bool get_answer = true; std::string terminalInput;
	while (get_answer) {

		std::cout << "Saving steps? (applied to the time vector, not the timestep): ";
		std::cin >> terminalInput;

		errorCheck(intCheck(get_answer, terminalInput, REINIT_ERR_MSG, 0, 1))
	}

	__storage_steps = std::stoi(terminalInput);
	#else
	__storage_steps = SAVING_STEPS;
	#endif

	const uintType savingArraySize = floor((double) __reinitialization_info.size() / __storage_steps) + 1;

	__simulation_storage.resize(__problem_domain.Total_Nodes() * savingArraySize);

	return 0;
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

		#if TERMINAL_INPUT_ALLOWED
		// Read number of samples from terminal
		bool get_answer = true;
		std::string inputTerminal;

		while (get_answer) {

			std::cout << "How many samples for parameter " << p + 1 << " ? ";
			std::cin >> inputTerminal;

			errorCheck(intCheck(get_answer, inputTerminal, DISTR_ERR_MSG, 0, 1))
		}
		__parameter_distributions[p].num_Samples = std::stoi(inputTerminal);
		#else
		__parameter_distributions[p].num_Samples = _DIST_N_SAMPLES[p];
		#endif
	}

	

	return 0;
};

// This function contains the most important function of them all: The full numerical method!
int16_t ivpSolver::evolvePDF(const cudaDeviceProp& D_Properties) {
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 	PARAMETERS

	intType totalSampleCount = 1, sum_sampleNum = 0;
	intType Samples_per_Param[PARAM_SPACE_DIMENSIONS];

	for (uintType i = 0; i < PARAM_SPACE_DIMENSIONS; i++) {
		uintType temp = __parameter_distributions[i].num_Samples;

		// Total samples from the parameter space discretization
		totalSampleCount *= temp;

		// Length of the compressed parameter array
		sum_sampleNum += temp;

		// Array storing the number of samples for each parameter
		Samples_per_Param[i] = temp;
	}

	// Full parameter array
	Param_pair* Parameter_Mesh = new Param_pair[sum_sampleNum];

	// Build the parameter mesh from previous info
	if (RANDOMIZE<PARAM_SPACE_DIMENSIONS>(Parameter_Mesh, __parameter_distributions) == -1) { delete[] Parameter_Mesh; return -1; }

	// Small vector containing the number of samples per parameter
	thrust::device_vector<intType>	D_sampVec(Samples_per_Param, Samples_per_Param + PARAM_SPACE_DIMENSIONS);

	// Parameter __problem_domain array (for the GPU)
	thrust::device_vector<Param_pair> D_Parameter_Mesh(Parameter_Mesh, Parameter_Mesh + sum_sampleNum);

	// auxiliary variable that will be used for ensemble mean computation
	floatType sum_sample_val = 0;
	for (uintType i = 0; i < totalSampleCount; i++) {
		Param_vec<PARAM_SPACE_DIMENSIONS> temp = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i, Parameter_Mesh, Samples_per_Param);
		sum_sample_val += temp.Joint_PDF;
	}

	// Memory management
	delete[] Parameter_Mesh;

	// Output to the CLI
	std::cout << "Total number of random samples: " << totalSampleCount << ".\n";

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PROBLEM DOMAIN AND INITIAL PDF

		// This Mesh will be defined by the support bounding box of the data.
	Mesh PDF_Support(IC_InfTVAL, IC_SupTVAL);

	// Make it square for AMR-purposes
	PDF_Support.Squarify();

	// Update the number of pts per dimension
	PDF_Support.Nodes_per_Dim = round((IC_SupTVAL[0] - IC_InfTVAL[0]) / __problem_domain.Discr_length());

	// PDF values at the fixed, high-res Mesh (CPU)
	thrust::host_vector<floatType> PDF_ProbDomain(__problem_domain.Total_Nodes(), 0);

	// initialize the Mesh and the PDF at the Mesh nodes (change so as to change the parameters as well)
	errorCheck(PDF_INITIAL_CONDITION(__problem_domain, PDF_ProbDomain, __initial_condition_distributions));

	// PDF values at fixed Grid Nodes (for the GPU)
	thrust::device_vector<floatType> D_PDF_ProbDomain = PDF_ProbDomain;

	// Particle positions (for the GPU)
	thrust::device_vector<Particle> D_Particle_Locations;

	// PDF value at Particle positions (for the GPU)
	thrust::device_vector<floatType> D_Particle_Values;

	// Now we make a slightly larger domain for the computations:
	Mesh Expanded_Domain;

	// Expand 40 nodes appears to be just fine (MAKE IT BETTER)
	const uint16_t expansion_nodes = 40;
	Expanded_Domain.Expand_From(__problem_domain, expansion_nodes);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES DEFINITION

	const uintType nrNodesPerFrame = __problem_domain.Total_Nodes();

	// Max memory to be used (in bytes). 95% just in case
	const uintType MAX_BYTES_USEABLE = 0.95 * (D_Properties.totalGlobalMem - nrNodesPerFrame * sizeof(floatType));

	// The following consts don't need an explanation
	const uintType	ConjGrad_MaxSteps = 1000;
	const floatType	RBF_SupportRadius = DISC_RADIUS * __problem_domain.Discr_length();

	// Full array storing appended particles for all parameter samples
	std::vector<Particle>	Full_Particle_Locations;
	std::vector<floatType>		Full_Particle_Values;

	thrust::device_vector<floatType> D_lambdas;
	thrust::device_vector<floatType>	D_Mat_Vals;
	thrust::device_vector<intType>	D_Mat_Indx;
	InterpHandle interpVectors;
	thrust::device_vector<Particle> D_fixedParticles;

	indicators::ProgressBar statusBar{ indicators::option::BarWidth{35},
										indicators::option::ForegroundColor{indicators::Color::yellow},
										indicators::option::ShowElapsedTime{true},
										indicators::option::ShowRemainingTime{false},
										indicators::option::PrefixText{"[INFO] Running..."},
										indicators::option::Start{"["},
										indicators::option::Fill{"*"},
										indicators::option::Lead{"*"},
										indicators::option::Remainder{"-"},
										indicators::option::End{"]"}, };

	// Resize the simulation logger
	__simulation_log.resize(__reinitialization_info.size() - 1);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// START OF SIMULATION STEPS AS SUCH

	// Simulation steps
	uint32_t simStepCount = 0;
	std::atomic<uint32_t> saveStepCount = 0;
	double t0, tF;

	// Aux variable to switch between step functions (Heaviside forcing) 
	uint32_t mode = 0;

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
	auto concurrentSaving = [&saveStepCount](const thrust::host_vector<floatType>& vSrc, std::vector<floatType>& vDst, const uintType simStepCount, const uintType __storage_steps, const uintType nrNodesPerFrame) {
		if ((simStepCount == 0 || simStepCount % __storage_steps == 0) && saveStepCount < vDst.size() / nrNodesPerFrame - 1) {
			// Store info in cumulative variable
			thrust::copy(vSrc.begin(), vSrc.end(), &vDst[saveStepCount * nrNodesPerFrame]);
			saveStepCount++;
		}
	};

	// IN THIS LINE WE START WITH THE ACTUAL ITERATIONS OF THE LIOUVILLE EQUATION
	while (simStepCount < __reinitialization_info.size() - 1) {

		// Save previous frame concurrently, so no time is used for this
		std::thread storeFrame_worker(concurrentSaving, std::ref(PDF_ProbDomain), std::ref(__simulation_storage), simStepCount, __storage_steps, nrNodesPerFrame);

		__simulation_log.LogFrames[simStepCount].simIteration = simStepCount;
		__simulation_log.LogFrames[simStepCount].simTime = __reinitialization_info[simStepCount].time;

		// select the first and last time value of the current iteration
		t0 = __reinitialization_info[simStepCount].time; tF = __reinitialization_info[simStepCount + 1].time;

		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// -------------------------- ADAPT. MESH REFINEMENT --------------------------------- //
		/////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////
		auto startTimeSeconds = std::chrono::high_resolution_clock::now();

		errorCheck(setInitialParticles(PDF_ProbDomain, D_PDF_ProbDomain,				// Initial, gridded PDF
			D_Particle_Values, D_Particle_Locations, 		// Output vectors that will give the relevant nodes
			__problem_domain, Expanded_Domain, PDF_Support));	// Domain information

		auto endTimeSeconds = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> durationSeconds = endTimeSeconds - startTimeSeconds;

		// Number of particles to advect
		uintType AMR_ActiveNodeCount = D_Particle_Locations.size();

		// To the Log file
		__simulation_log.LogFrames[simStepCount].log_AMR_Time = durationSeconds.count();
		__simulation_log.LogFrames[simStepCount].log_AMR_RelevantParticles = AMR_ActiveNodeCount;

		#if ERASE_dPDF
		// Clear the GPU-stored PDF for better memory availability
		D_PDF_ProbDomain.clear();
		#endif

		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// -------------------------- PT. SEARCH --------------------------------------------- //
		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////

		// Maximum neighbors to search. Diameter number of points powered to the dimension
		uintType MaxNeighborNum = round(fmin(pow(2 * round(DISC_RADIUS) + 1, PHASE_SPACE_DIMENSIONS), AMR_ActiveNodeCount));

		// Compressed COO-style indexing of the sparse interpolation matrix
		D_Mat_Indx.resize(MaxNeighborNum * AMR_ActiveNodeCount);
		thrust::fill(D_Mat_Indx.begin(), D_Mat_Indx.end(), -1);
		// Sparse interpolation matrix values
		D_Mat_Vals.resize(MaxNeighborNum * AMR_ActiveNodeCount);
		thrust::fill(D_Mat_Vals.begin(), D_Mat_Vals.end(), 0);


		startTimeSeconds = std::chrono::high_resolution_clock::now();

		errorCheck(particleNeighborSearch(D_Particle_Locations,
			D_Particle_Values,
			D_Mat_Indx,
			D_Mat_Vals,
			AMR_ActiveNodeCount,
			MaxNeighborNum,
			PDF_Support,
			RBF_SupportRadius));

		endTimeSeconds = std::chrono::high_resolution_clock::now();
		durationSeconds = endTimeSeconds - startTimeSeconds;

		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// -------------------------- INTERPOLATION ------------------------------------------ //
		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// Declare the solution of the interpolation vector (weights of the RBF functions)
		D_lambdas.resize(AMR_ActiveNodeCount);
		thrust::fill(D_lambdas.begin(), D_lambdas.end(), 0);

		interpVectors.resize(AMR_ActiveNodeCount);

		startTimeSeconds = std::chrono::high_resolution_clock::now();
		int32_t iterations = CONJUGATE_GRADIENT_SOLVE(D_lambdas,
			D_Mat_Indx,
			D_Mat_Vals,
			D_Particle_Values,
			interpVectors,
			AMR_ActiveNodeCount,
			MaxNeighborNum,
			ConjGrad_MaxSteps,
			TOLERANCE_ConjGrad);
		if (iterations == -1) { std::cout << "Convergence failure.\n"; break; }
		endTimeSeconds = std::chrono::high_resolution_clock::now();
		durationSeconds = endTimeSeconds - startTimeSeconds;


		// To the Log file
		__simulation_log.LogFrames[simStepCount].log_Interpolation_Time = durationSeconds.count();
		__simulation_log.LogFrames[simStepCount].log_Interpolation_Iterations = iterations;

		#if ERASE_auxVectors == true
		// Clear the vectors to save memory
		D_Mat_Indx.clear();
		D_Mat_Vals.clear();
		#endif

		D_Particle_Values = D_lambdas;

		if (__reinitialization_info[simStepCount].impulse) {
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- DELTA/HEAVISIDE IMPULSE TERMS -------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
		#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-floatType IMPULSE!

			startTimeSeconds = std::chrono::high_resolution_clock::now();

			errorCheck(IMPULSE_TRANSFORM_PDF(D_PDF_ProbDomain,
				D_Particle_Locations,
				D_Particle_Values,
				__reinitialization_info[simStepCount],
				jumpCount,
				__problem_domain,
				Expanded_Domain,
				PDF_Support));

			endTimeSeconds = std::chrono::high_resolution_clock::now();
			durationSeconds = endTimeSeconds - startTimeSeconds;

			// Enter the information into the log information
			__simulation_log.LogFrames[simStepCount].log_Advection_Time = durationSeconds.count();
			__simulation_log.LogFrames[simStepCount].log_Advection_TotalParticles = durationSeconds.count();

			jumpCount++;

			// Send back to CPU
			PDF_ProbDomain = D_PDF_ProbDomain;


		#elif(IMPULSE_TYPE == 2)
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// ----------------------------- HEAVISIDE/STEP IMPULSE ------------------------------ //
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
			mode++;

		#elif(IMPULSE_TYPE != 0)
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// -------------------------- ERROR:  UNDEFINED IMPULSE floatType ------------------------- //
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
			std::cout << "Error in 'Dynamics.cuh'. You are choosing an unavailable option. Go back to 'Case_definition.cuh' and re-check options for IMPULSE_TYPE.\n";
			error_check = -1;
			break;

		#endif
		}
		else {
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- SMOOTH PARTICLE INTEGRATION ---------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////

			// Max. memory requirements for next step
			const uintType Bytes_per_sample = AMR_ActiveNodeCount * (sizeof(floatType) * 2 + sizeof(Particle));

			// Set number of random samples to work with at the same time
			uintType Samples_PerBlk = fmin((uintType)totalSampleCount, MAX_BYTES_USEABLE / Bytes_per_sample);

			// Number of blocks to simulate
			uintType total_simulation_blocks = ceil((double)totalSampleCount / Samples_PerBlk);

			// For correct reinitialization
			D_fixedParticles = D_Particle_Locations;

			for (uintType b = 0; b < total_simulation_blocks; b++) {

				// Parameter sample offset init. and final to account for the block position
				uintType Sample_idx_offset_init = b * Samples_PerBlk;
				uintType Sample_idx_offset_final = fmin((b + 1) * Samples_PerBlk, totalSampleCount);

				// Actual number of samples in current block
				Samples_PerBlk = Sample_idx_offset_final - Sample_idx_offset_init;

				// Total AMR-activated nodes in the current block
				uintType ActiveNodes_PerBlk = Samples_PerBlk * AMR_ActiveNodeCount;

				D_Particle_Locations.resize(ActiveNodes_PerBlk);
				D_Particle_Values.resize(ActiveNodes_PerBlk);

				for (uintType k = 0; k < Samples_PerBlk; k++) {
					thrust::copy(thrust::device, &D_fixedParticles[0], &D_fixedParticles[AMR_ActiveNodeCount],
						&D_Particle_Locations[k * AMR_ActiveNodeCount]);

					thrust::copy(thrust::device, &D_lambdas[0], &D_lambdas[AMR_ActiveNodeCount],
						&D_Particle_Values[k * AMR_ActiveNodeCount]);
				}

				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				// -------------------------- POINT ADVECTION ---------------------------------------- //
				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				uint16_t Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
				uintType Blocks = floor((double)(ActiveNodes_PerBlk - 1) / Threads) + 1;

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				ODE_INTEGRATE << <Blocks, Threads >> > (
					rpc(D_Particle_Locations, 0),
					rpc(D_Particle_Values, 0),
					rpc(D_Parameter_Mesh, Sample_idx_offset_init),
					rpc(D_sampVec, 0),
					t0,
					__delta_t,
					tF,
					AMR_ActiveNodeCount,
					Samples_PerBlk,
					mode,
					rpc(Extra_Parameter, 0),
					__problem_domain);
				gpuError_Check(cudaDeviceSynchronize());
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				durationSeconds = endTimeSeconds - startTimeSeconds;

				// To the Log file
				__simulation_log.LogFrames[simStepCount].log_Advection_Time = durationSeconds.count();
				__simulation_log.LogFrames[simStepCount].log_Advection_TotalParticles = ActiveNodes_PerBlk;

				PDF_Support.Update_boundingBox(D_Particle_Locations);

				// COMPUTE THE SOLUTION "PROJECTION" INTO THE L1 SUBSPACE. THIS WAY, REINITIALIZATION CONSERVES VOLUME (=1)
				if (PHASE_SPACE_DIMENSIONS < 5) {
					floatType temp = thrust::reduce(thrust::device, D_Particle_Values.begin(), D_Particle_Values.end());
					thrust::transform(D_Particle_Values.begin(), D_Particle_Values.end(), D_Particle_Values.begin(), Samples_PerBlk / temp * _1);
				}

				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				// -------------------------- REINITIALIZATION --------------------------------------- //
				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////

				#if ERASE_dPDF
				D_PDF_ProbDomain.resize(nrNodesPerFrame, 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
				#else
				thrust::fill(thrust::device, D_PDF_ProbDomain.begin(), D_PDF_ProbDomain.end(), 0);
				#endif

				Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
				Blocks = floor((double)(ActiveNodes_PerBlk - 1) / Threads) + 1;

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				RESTART_GRID_FIND_GN << < Blocks, Threads >> > (rpc(D_Particle_Locations, 0),
					rpc(D_PDF_ProbDomain, 0),
					rpc(D_Particle_Values, 0),
					rpc(D_Parameter_Mesh, 0),
					rpc(D_sampVec, 0),
					RBF_SupportRadius,
					AMR_ActiveNodeCount,
					Samples_PerBlk,
					Sample_idx_offset_init,
					__problem_domain,
					Expanded_Domain);
				gpuError_Check(cudaDeviceSynchronize());
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				durationSeconds = endTimeSeconds - startTimeSeconds;

				// To the Log file
				__simulation_log.LogFrames[simStepCount].log_Reinitialization_Time = durationSeconds.count();
			}

			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- STORE PDF INTO OUTPUT ARRAY ---------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////

			// Correction of any possible negative PDF values
			uintType Threads = fmin(THREADS_P_BLK, nrNodesPerFrame / ELEMENTS_AT_A_TIME);
			uintType Blocks = floor((double)(nrNodesPerFrame / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;

			CORRECTION << <Blocks, Threads >> > (rpc(D_PDF_ProbDomain, 0), nrNodesPerFrame);
			gpuError_Check(cudaDeviceSynchronize());

			// Divide by the sum of the values of the parameter mesh to obtain the weighted mean
			thrust::transform(D_PDF_ProbDomain.begin(), D_PDF_ProbDomain.end(), D_PDF_ProbDomain.begin(), 1.0f / sum_sample_val * _1); // we use the thrust::placeholders here (@ the last input argument)

			// Send back to CPU
			PDF_ProbDomain = D_PDF_ProbDomain;

		}

		// Upadte simulation step
		simStepCount++;

		storeFrame_worker.join();

		statusBar.set_option(indicators::option::PostfixText{ "Iterations: " + std::to_string(simStepCount) + "/" + std::to_string(__reinitialization_info.size() - 1) });
		statusBar.set_progress((float)simStepCount / (__reinitialization_info.size() - 1) * 100);
	}

	thrust::copy(PDF_ProbDomain.begin(), PDF_ProbDomain.end(), &__simulation_storage[saveStepCount * nrNodesPerFrame]);

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

	const uintType nrNodesPerFrame = __problem_domain.Total_Nodes();

	const uint64_t MEM_2_STORE = __simulation_storage.size() * sizeof(float);

	uintType number_of_frames_needed = floor((double) __simulation_storage.size() / nrNodesPerFrame);

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

				file1 << nrNodesPerFrame << "," << __problem_domain.Nodes_per_Dim << ",";

				for (uintType d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
					file1 << __problem_domain.Boundary_inf.dim[d] << "," << __problem_domain.Boundary_sup.dim[d] << ",";
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

			file1 << nrNodesPerFrame << "," << __problem_domain.Nodes_per_Dim << ",";

			for (uintType d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
				file1 << __problem_domain.Boundary_inf.dim[d] << "," << __problem_domain.Boundary_sup.dim[d] << ",";
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
