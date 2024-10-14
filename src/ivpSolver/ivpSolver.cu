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


// Set the default constructor. Parametric constructor won't be required!!
ivpSolver::ivpSolver() {};
ivpSolver::~ivpSolver() {};

// This method is used to build the domain in which the simulation will take place
int16_t ivpSolver::buildDomain() {

	// This variable represents the problem domain, which is NOT going to be the one used for computations
	m_problemDomain.setNodesPerDimension(pow(2, FINEST_DISCR_LVL));

	return 0;
};

// Use the time information 
int16_t ivpSolver::buildTimeVec() {

	errorCheck(BuildTimeVector(m_reinitializationInfo, __delta_t, __reinitialization_steps))
	m_storageSteps = SAVING_STEPS;

	const uintType savingArraySize = ceil((double) m_reinitializationInfo.size() / m_storageSteps);

	m_simulationStorage.resize(m_problemDomain.total_nodes() * savingArraySize);

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
		m_parameterDistributions[p].Name = _DIST_NAMES[p];					// N, G or U distributions
		m_parameterDistributions[p].isTruncated = _DIST_isTRUNC[p];		// TRUNCATED?
		m_parameterDistributions[p].trunc_interval[0] = _DIST_InfTVAL[p];	// min of trunc. interval
		m_parameterDistributions[p].trunc_interval[1] = _DIST_SupTVAL[p]; 	// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
		m_parameterDistributions[p].params[0] = _DIST_MEAN[p];				// mean
		m_parameterDistributions[p].params[1] = _DIST_STD[p];				// std
		m_parameterDistributions[p].num_Samples = _DIST_N_SAMPLES[p];		// Number of samples per parameter (assumed that parameters are statistically independent)
	}

	return EXIT_SUCCESS;
};

// This function contains the most important function of them all: The full numerical method!
int16_t ivpSolver::evolvePDF() {
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 	PARAMETERS
	uint32_t total_sample_count = 1, sum_samples_count = 0;
	intType samples_per_parameter[PARAM_SPACE_DIMENSIONS];

	for (uintType i = 0; i < PARAM_SPACE_DIMENSIONS; i++) {
		uintType temp = m_parameterDistributions[i].num_Samples;

		// Total samples from the parameter space discretization
		total_sample_count *= temp;

		// Length of the compressed parameter array
		sum_samples_count += temp;

		// Array storing the number of samples for each parameter
		samples_per_parameter[i] = temp;
	}

	// Full parameter array
	std::vector<parameterPair> parameterMesh(sum_samples_count);

	// Build the parameter mesh from previous info
	if (RANDOMIZE<PARAM_SPACE_DIMENSIONS>(parameterMesh, m_parameterDistributions)) { return EXIT_FAILURE; }
	
	deviceUniquePtr<parameterPair> parameterMesh_dvc(sum_samples_count, parameterPair());
	gpu_device.memCpy_hst2dvc(parameterMesh_dvc.get(), parameterMesh.data(), parameterMesh_dvc.size_bytes());

	deviceUniquePtr<intType> samplesPerParameter_dvc(PARAM_SPACE_DIMENSIONS);
	gpu_device.memCpy_hst2dvc(samplesPerParameter_dvc.get(), &samples_per_parameter[0], samplesPerParameter_dvc.size_bytes());

	// auxiliary variable that will be used for ensemble mean computation
	floatType sum_sample_val = 0;
	for (uintType i = 0; i < total_sample_count; i++) {
		Param_vec<PARAM_SPACE_DIMENSIONS> temp = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i, parameterMesh.data(), samples_per_parameter);
		sum_sample_val += temp.Joint_PDF;
	}

	// Output to the CLI
	mainTerminal.print_message("Total number of random samples: " + std::to_string(total_sample_count));

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PROBLEM DOMAIN AND INITIAL PDF

	//// This cartesianMesh will be defined by the support bounding box of the data.
	m_particleBoundingBox = m_problemDomain;

	// PDF values at the fixed, high-res cartesianMesh (CPU)
	std::unique_ptr<floatType[]> pdfValuesAtProblemDomain = std::make_unique<floatType[]>(m_problemDomain.total_nodes());

	// initialize the cartesianMesh and the PDF at the cartesianMesh nodes (change so as to change the parameters as well)
	errorCheck(PDF_INITIAL_CONDITION(m_problemDomain, pdfValuesAtProblemDomain.get(), __initial_condition_distributions));

	// Pass the pdfValuesAtProblemDomain to the gpu:
	deviceUniquePtr<floatType> pdfValuesAtProblemDomain_dvc(m_problemDomain.total_nodes());
	gpu_device.memCpy_hst2dvc(
		pdfValuesAtProblemDomain_dvc.get(),
		pdfValuesAtProblemDomain.get(),
		m_problemDomain.total_nodes() * sizeof(floatType)
	);

	// Now we make a slightly larger domain for the computations:
	cartesianMesh Expanded_Domain;

	// Expand 40 nodes appears to be just fine (MAKE IT BETTER)
	const uint16_t expansion_nodes = 40;
	Expanded_Domain.Expand_From(m_problemDomain, expansion_nodes);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES DEFINITION

	const uintType nrNodesPerFrame = m_problemDomain.total_nodes();

	// Max memory to be used (in bytes). 95% just in case
	const uintType MAX_BYTES_USEABLE = 0.95 * (gpu_device.deviceProperties.totalGlobalMem - nrNodesPerFrame * sizeof(floatType));

	// The following consts don't need an explanation
	const floatType	RBF_SupportRadius = DISC_RADIUS * m_problemDomain.discr_length();

	// Resize the simulation logger
	m_simulationLog.resize(m_reinitializationInfo.size() - 1);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// START OF SIMULATION STEPS AS SUCH

	// Simulation steps
	uintType iterationCount = 0;
	// Saved frames
	std::atomic<uintType> currentlySavedFrames = 0;	// We need atomic type because we'll be saving in a separate thread concurrently!

	// Aux variable to switch between step functions (Heaviside forcing) 
	uint16_t mode = 0;

	// IF THERE ARE DELTA TERMS
	#if IMPULSE_TYPE == 1
	uintType jumpCount = 0;	// auxiliary variable to know how many delta jumpCounts have passed
	#endif

	// TODO: FIX THIS APPROPRIATELY
	#if INCLUDE_XTRA_PARAMS
	deviceUniquePtr<double>	extraParameters_dvc(XTRA_PARAM_LENGTH);
	thrust::copy(&XTRA_PARAM[0], &XTRA_PARAM[XTRA_PARAM_LENGTH], extraParameters_dvc.begin());
	#else
	deviceUniquePtr<double> extraParameters_dvc;
	#endif

	// Define concurrent saving lambda function
	auto concurrentSaving = [&currentlySavedFrames](const floatType* vSrc, std::vector<floatType>& vDst, const uintType iterationCount, const uintType m_storageSteps, const uintType nrNodesPerFrame) {
		if ((iterationCount == 0 || iterationCount % m_storageSteps == 0) && currentlySavedFrames < vDst.size() / nrNodesPerFrame - 1) {
			// Store info in cumulative variable
			std::copy(vSrc, vSrc + currentlySavedFrames * nrNodesPerFrame, vDst.begin());
			currentlySavedFrames++;
		}
	};

	// IN THIS LINE WE START WITH THE ACTUAL ITERATIONS OF THE LIOUVILLE EQUATION
	while (iterationCount < m_reinitializationInfo.size() - 1) {

		// Save previous frame concurrently, so no time is used for this
		std::thread storeFrame_worker(concurrentSaving, pdfValuesAtProblemDomain.get(), std::ref(m_simulationStorage), iterationCount, m_storageSteps, nrNodesPerFrame);

		m_simulationLog.LogFrames[iterationCount].simIteration = iterationCount;
		m_simulationLog.LogFrames[iterationCount].simTime = m_reinitializationInfo[iterationCount].time;

		// select the first and last time value of the current iteration
		double t0 = m_reinitializationInfo[iterationCount].time; double tF = m_reinitializationInfo[iterationCount + 1].time;

		// Full array storing appended particles for all parameter samples. Use std::vector because it changes quite often!
		deviceUniquePtr<Particle>	particleLocations_dvc;
		deviceUniquePtr<floatType>	particleValues_dvc;

		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// -------------------------- ADAPT. MESH REFINEMENT --------------------------------- //
		/////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////
		auto startTimeSeconds = std::chrono::high_resolution_clock::now();

		try{
			setInitialParticles(
				pdfValuesAtProblemDomain_dvc.get(),
				particleLocations_dvc,
				particleValues_dvc,
				m_particleBoundingBox,
				m_problemDomain
			);
		}
		catch (const std::exception& except) {
			std::cerr << "Exception caught at setInitialParticles. Code: " << except.what() << std::endl;

			storeFrame_worker.join();	// Make sure that the dispatched thread is not working when we return from the function
			return EXIT_FAILURE;
		}

		auto endTimeSeconds = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> durationSeconds = endTimeSeconds - startTimeSeconds;

		// Number of particles to advect
		uintType AMR_ActiveNodeCount = particleLocations_dvc.size_count();

		// To the Log file
		m_simulationLog.LogFrames[iterationCount].log_AMR_Time = durationSeconds.count();
		m_simulationLog.LogFrames[iterationCount].log_AMR_RelevantParticles = AMR_ActiveNodeCount;

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
		deviceUniquePtr<int64_t> matrixIndex_dvc(MaxNeighborNum * AMR_ActiveNodeCount, (int64_t)-1);
		deviceUniquePtr<floatType> matrixValues_dvc(MaxNeighborNum * AMR_ActiveNodeCount, (floatType)0.0);

		startTimeSeconds = std::chrono::high_resolution_clock::now();
		errorCheck(particleNeighborSearch(
			particleLocations_dvc,
			particleValues_dvc,		// Not actually used for now
			matrixIndex_dvc,
			matrixValues_dvc,
			AMR_ActiveNodeCount,
			MaxNeighborNum,
			m_particleBoundingBox,
			RBF_SupportRadius));

		endTimeSeconds = std::chrono::high_resolution_clock::now();
		durationSeconds = endTimeSeconds - startTimeSeconds;

		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// -------------------------- INTERPOLATION ------------------------------------------ //
		/////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		// Declare the solution of the interpolation vector (weights of the RBF functions)
		deviceUniquePtr<floatType> basisWeightsLambdas_dvc(AMR_ActiveNodeCount, (floatType)0);
		ConjugateGradientEngine interpolationEngine(AMR_ActiveNodeCount);

		startTimeSeconds = std::chrono::high_resolution_clock::now();
		intType iterations = interpolationEngine.execute(
			basisWeightsLambdas_dvc,
			particleValues_dvc,
			matrixIndex_dvc,
			matrixValues_dvc,
			MaxNeighborNum
		);
		if (iterations == -1) { std::cout << "Convergence failure.\n"; break; }
		endTimeSeconds = std::chrono::high_resolution_clock::now();
		durationSeconds = endTimeSeconds - startTimeSeconds;


		// To the Log file
		m_simulationLog.LogFrames[iterationCount].log_Interpolation_Time = durationSeconds.count();
		m_simulationLog.LogFrames[iterationCount].log_Interpolation_Iterations = iterations;

		#if ERASE_auxVectors == true
		// Clear the vectors to save memory
		matrixIndex_dvc.clear();
		matrixValues_dvc.clear();
		#endif

		// Overwrite the particle values with the lambdas from the RBF interpolation procedure
		gpu_device.memCpy_dvc2dvc(
			particleValues_dvc.get(), basisWeightsLambdas_dvc.get(), basisWeightsLambdas_dvc.size_bytes()
		);

		if (m_reinitializationInfo[iterationCount].impulse) {
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- DELTA/HEAVISIDE IMPULSE TERMS -------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
		#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-floatType IMPULSE!

			startTimeSeconds = std::chrono::high_resolution_clock::now();

			errorCheck(IMPULSE_TRANSFORM_PDF(D_PDF_ProbDomain,
				particleLocations_dvc,
				particleValues_dvc,
				m_reinitializationInfo[iterationCount],
				jumpCount,
				m_problemDomain,
				Expanded_Domain,
				m_particleBoundingBox));

			endTimeSeconds = std::chrono::high_resolution_clock::now();
			durationSeconds = endTimeSeconds - startTimeSeconds;

			// Enter the information into the log information
			m_simulationLog.LogFrames[iterationCount].log_Advection_Time = durationSeconds.count();
			m_simulationLog.LogFrames[iterationCount].log_Advection_TotalParticles = durationSeconds.count();

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
			continue;
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
			uintType Samples_PerBlk = fmin((uintType)total_sample_count, MAX_BYTES_USEABLE / Bytes_per_sample);

			// Number of blocks to simulate
			uintType total_simulation_blocks = ceil((double)total_sample_count / Samples_PerBlk);

			for (uintType b = 0; b < total_simulation_blocks; b++) {

				// Parameter sample offset init. and final to account for the block position
				uintType Sample_idx_offset_init = b * Samples_PerBlk;
				uintType Sample_idx_offset_final = fmin((b + 1) * Samples_PerBlk, total_sample_count);

				// Actual number of samples in current block
				Samples_PerBlk = Sample_idx_offset_final - Sample_idx_offset_init;

				// Total AMR-activated nodes in the current block
				uintType ActiveNodes_PerBlk = Samples_PerBlk * AMR_ActiveNodeCount;

				deviceUniquePtr<Particle> fullParticleLocations_dvc(ActiveNodes_PerBlk, Particle());
				deviceUniquePtr<floatType> fullBasisWeightsLambdas_dvc(ActiveNodes_PerBlk, (floatType) 0);

				for (uintType k = 0; k < Samples_PerBlk; k++) {
					gpu_device.memCpy_dvc2dvc(
						fullParticleLocations_dvc.get(k * AMR_ActiveNodeCount), particleLocations_dvc.get(),
						particleLocations_dvc.size_bytes()
					);
					gpu_device.memCpy_dvc2dvc(
						fullBasisWeightsLambdas_dvc.get(k * AMR_ActiveNodeCount), particleValues_dvc.get(),
						particleValues_dvc.size_bytes()
					);
				}


				// TODO: ME HE QUEDADO AQUÍ!!!!!
				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				// -------------------------- POINT ADVECTION ---------------------------------------- //
				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				uint16_t Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
				uint64_t Blocks = floor((double)(ActiveNodes_PerBlk - 1) / Threads) + 1;

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				try{
					gpu_device.launchKernel(Blocks, Threads, characteristicIntegrator{
						fullParticleLocations_dvc.get(),
						fullBasisWeightsLambdas_dvc.get(),
						parameterMesh_dvc.get(Sample_idx_offset_init),
						samplesPerParameter_dvc.get(),
						t0,
						__delta_t,
						tF,
						AMR_ActiveNodeCount,
						total_sample_count,
						mode,
						extraParameters_dvc.get(),
						m_problemDomain
						});
				}
				catch (const std::exception& except) {
					std::cerr << "Exception caught at characteristic integration. Code: " << except.what() << std::endl;
					return EXIT_FAILURE;
				}
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				durationSeconds = endTimeSeconds - startTimeSeconds;

				// To the Log file
				m_simulationLog.LogFrames[iterationCount].log_Advection_Time = durationSeconds.count();
				m_simulationLog.LogFrames[iterationCount].log_Advection_TotalParticles = ActiveNodes_PerBlk;

				m_particleBoundingBox.update_bounding_box(particleLocations_dvc);

				// COMPUTE THE SOLUTION "PROJECTION" INTO THE L1 SUBSPACE. THIS WAY, REINITIALIZATION CONSERVES VOLUME (=1)
				if (PHASE_SPACE_DIMENSIONS < 5) {
					L1normalizeLambdas(fullBasisWeightsLambdas_dvc.get(), Samples_PerBlk, ActiveNodes_PerBlk);
				}

				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				// -------------------------- REINITIALIZATION --------------------------------------- //
				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				// Restart the values of the PDF at the problem domain
				pdfValuesAtProblemDomain_dvc.set_init_values((floatType)0);

				Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
				Blocks = floor((double)(ActiveNodes_PerBlk - 1) / Threads) + 1;

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				try{
					gpu_device.launchKernel(Blocks, Threads, remeshParticles<parameterPair>{
						fullParticleLocations_dvc.get(),
							pdfValuesAtProblemDomain_dvc.get(),
							fullBasisWeightsLambdas_dvc.get(),
							parameterMesh_dvc.get(),
							samplesPerParameter_dvc.get(),
							RBF_SupportRadius,
							AMR_ActiveNodeCount,
							Samples_PerBlk,
							Sample_idx_offset_init,
							m_problemDomain,
							Expanded_Domain
					});
				}
				catch (const std::exception& except) {
					std::cerr << "Exception caught at particle remeshing. Code: " << except.what() << std::endl;
					return EXIT_FAILURE;
				}
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				durationSeconds = endTimeSeconds - startTimeSeconds;

				// To the Log file
				m_simulationLog.LogFrames[iterationCount].log_Reinitialization_Time = durationSeconds.count();
			}

			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- STORE PDF INTO OUTPUT ARRAY ---------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////

			// Correction of any possible negative PDF values
			uint16_t Threads = fmin(THREADS_P_BLK, nrNodesPerFrame / ELEMENTS_AT_A_TIME);
			uint64_t Blocks = floor((double)(nrNodesPerFrame / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;
			try{
				gpu_device.launchKernel(Blocks, Threads, correctNegativeValues<floatType>{
					pdfValuesAtProblemDomain_dvc.get(),
					nrNodesPerFrame
				});
			}
			catch (const std::exception& except) {
				std::cerr << "Exception cauhgt correcting PDF negative values. Code: " << except.what() << std::endl;
				return EXIT_FAILURE;
			}

			// Divide by the sum of the values of the parameter mesh to obtain the weighted mean
			try{
				gpu_device.transformMultiply<floatType>(
					pdfValuesAtProblemDomain_dvc.get(), 1.0f / sum_sample_val, nrNodesPerFrame);
			}
			catch (const std::exception& except) {
				std::cerr << "Exception caught at multiply per value. Code: " << except.what() << std::endl;
				return EXIT_FAILURE;
			}
			
			// Send back to CPU
			gpu_device.memCpy_dvc2hst(
				pdfValuesAtProblemDomain.get(),
				pdfValuesAtProblemDomain_dvc.get(),
				pdfValuesAtProblemDomain_dvc.size_bytes()
			);
		}

		// Upadte simulation step
		iterationCount++;

		storeFrame_worker.join();

		m_terminal.update_simulation_status(iterationCount, m_reinitializationInfo.size() - 1);
	}

	/*thrust::copy(PDF_ProbDomain.begin(), PDF_ProbDomain.end(), &m_simulationStorage[currentlySavedFrames * nrNodesPerFrame]);*/
	std::copy(pdfValuesAtProblemDomain.get(), pdfValuesAtProblemDomain.get() + nrNodesPerFrame, &m_simulationStorage[currentlySavedFrames * nrNodesPerFrame]);

	m_terminal.simulation_completed();

	std::string log_filename{CASE};
	log_filename += "_log_file"; 
	m_simulationLog.writeSimulationLog_toFile(log_filename);
	// Exit current function
	return EXIT_SUCCESS;
};

int16_t ivpSolver::writeFramesToFile(const double& simulationDuration) {
	// We have added the capability of automatically detecting the number of 1 GB files where we can store the simulation output
	bool saving_active = true;		// see if saving is still active
	int16_t error_check = 0;

	const uintType nrNodesPerFrame = m_problemDomain.total_nodes();

	const uint64_t MEM_2_STORE = m_simulationStorage.size() * sizeof(float);

	uintType number_of_frames_needed = m_simulationStorage.size() / nrNodesPerFrame;

	uint64_t max_frames_file = MAX_FILE_SIZE_B / nrNodesPerFrame / sizeof(float);
	uintType number_of_files_needed = floor((double)(number_of_frames_needed - 1) / max_frames_file) + 1;

	std::cout << "\nSimulation time: " << simulationDuration << " seconds. ";

	if (number_of_files_needed == 0) {
		std::cout << "There has been a problem. No memory written. Exiting simulation.\n";
		saving_active = false;
		error_check = -1;
	}

	while (saving_active) {
		
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

			file1 << nrNodesPerFrame << "," << m_problemDomain.nodes_per_dim() << ",";

			for (uintType d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
				file1 << m_problemDomain.boundary_inf().dim[d] << "," << m_problemDomain.boundary_sup().dim[d] << ",";
			}
			for (uint16_t d = 0; d < PARAM_SPACE_DIMENSIONS; d++) {
				file1 << m_parameterDistributions[d].num_Samples << ",";
			}
			file1 << simulationDuration << "\n";

			for (uintType i = k * max_frames_file + frames_init; i < k * max_frames_file + frames_in_file + frames_init - 1; i++) {
				file1 << m_reinitializationInfo[i * m_storageSteps].time << ",";
			}

			// Store the last element of the time vector, even if it is not proportional to the number of saving steps
			if (number_of_files_needed == 1){
				file1 << m_reinitializationInfo.back().time;
			}
			else{
				file1 << m_reinitializationInfo[(k * max_frames_file + frames_in_file + frames_init - 1) * m_storageSteps].time;
			}

			file1.close();

			// SIMULATION OUTPUT
			relative_pth = source_path + "/output/" + CASE;
			relative_pth.append("_Mean_PDFs_");
			relative_pth.append(temp_str);
			relative_pth.append(".bin");

			std::ofstream myfile(relative_pth, std::ios::out | std::ios::binary);
			assert(myfile.is_open());

			myfile.write((char*)&m_simulationStorage[(k * max_frames_file + frames_init) * nrNodesPerFrame], sizeof(floatType) * frames_in_file * nrNodesPerFrame);
			myfile.close();

			std::string temp_output_str = "Simulation output file " + std::to_string(k) + " completed!";
			m_terminal.print_message(temp_output_str);
		}
		saving_active = false;
	}
	return error_check;
};
