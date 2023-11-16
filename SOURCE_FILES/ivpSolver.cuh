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

#ifndef __IVPSOLVER_CUH__
#define __IVPSOLVER_CUH__

#include "Constants.cuh"
#include "Adapt_Mesh.cuh"
#include "Domain.cuh"
#include "Sim_data.cuh"
#include "Probability.cuh"
#include "Simulation_parameters.cuh"
#include "Interpolation.cuh"
#include "Impulse_transformations.cuh" // STILL GOTTA FIX THIS!
#include "Integrator.cuh"


namespace ivpSolver{

class ivpSolver{
public:
	grid Problem_Domain;
	
	Distributions IC_Distributions[PHASE_SPACE_DIMENSIONS];
	Distributions Parameter_Distributions[PARAM_SPACE_DIMENSIONS];
	
	std::vector<Time_instants> time_vector;
	double deltaT;
	int32_t ReinitSteps;

	std::vector<TYPE> storeFrames;

	// Set the default constructor. Parametric constructor won't be needed!
	ivpSolver(){}

// Methods:
public:
	// This method is used to build the domain in which the simulation will take place
	int16_t buildDomain(){

		bool getAnswer=true;

		while(getAnswer){

			std::string inputTerminal;
			std::cout << "Finest level in the domain?: ";
			std::cin >> inputTerminal;

			if(!isNumeric(inputTerminal)){std::cout << "Error: Non-numerical values not allowed. ";}
			else{
				int16_t LvlFine = std::stoi(inputTerminal);
				if(LvlFine == -1){
					std::cout << "Error building Domain. Exiting simulation\n";
					return -1;
				}
				else if(LvlFine < -1 || LvlFine == 0){
					std::cout << "You must choose a STRICTLY positive integer!. Try again...\n";
				}
				else{
					getAnswer = false;
				}
			}
		}
		
		// This variable represents the problem domain, which is NOT going to be the one used for computations
		Problem_Domain.Nodes_per_Dim = pow(2, LvlFine);

		return 0;
	}

	// Use the time information 
	int16_t buildTimeVec(){

		if (BuildTimeVector(time_vector, deltaT, ReinitSteps) == -1){std::cout << "Error building the time vector. Exiting simulation\n"; return -1;}

		storeFrames.resize(Problem_Domain.Total_Nodes() * time_vector.size());

		return 0;
	}

	// Build the Distribution information for all random parameters
	int16_t getDistributions(){

		// Read initial probability distribution information from the Case_definition file
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			IC_Distributions[d].Name 				= IC_NAMES[d];
			IC_Distributions[d].isTruncated 		= IC_isTRUNC[d];
			IC_Distributions[d].trunc_interval[0] 	= IC_InfTVAL[d];
			IC_Distributions[d].trunc_interval[1] 	= IC_SupTVAL[d];
			IC_Distributions[d].params[0] 			= IC_MEAN[d];
			IC_Distributions[d].params[1] 			= IC_STD[d];
			IC_Distributions[d].num_Samples 		= -1;		// This value is just for information. Samples are not chosen here
		}

		// Read parameter's probability distribution information from the Case_definition file
		for (uint16_t p = 0; p < PARAM_SPACE_DIMENSIONS; p++){
			Parameter_Distributions[p].Name  			 = _DIST_NAMES[p];		// N, G or U distributions
			Parameter_Distributions[p].isTruncated  	 = _DIST_isTRUNC[p];		// TRUNCATED?
			Parameter_Distributions[p].trunc_interval[0] = _DIST_InfTVAL[p];		// min of trunc. interval
			Parameter_Distributions[p].trunc_interval[1] = _DIST_SupTVAL[p]; 	// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
			Parameter_Distributions[p].params[0] 		 = _DIST_MEAN[p];		// mean
			Parameter_Distributions[p].params[1] 		 = _DIST_STD[p];			// std

			// Read number of samples from terminal
			bool get_answer = true;
			INT temp = 1;

			while (get_answer) {
				std::string inputTerminal;

				std::cout << "How many samples for parameter " << p + 1 << " ? ";
				std::cin >> inputTerminal;

				if(!isNumeric(inputTerminal)){std::cout << "Error: Non-numerical inputs not allowed. ";}
				else{

					temp = std::stoi(inputTerminal);

					if (temp == -1) {
						std::cout << "Error defining distributions. Exiting simulation.\n"; return -1;
					}

					if (temp == 0) {
						std::cout << "At least 1 sample must be selected. ";
					}
					else {
						get_answer = false;
					} 
				}
			}
			Parameter_Distributions[p].num_Samples = temp;
		}
		return 0;
	}
	
	// This function contains the most important function of them all: The full numerical method!
	int16_t EvolvePDF(const cudaDeviceProp* D_Properties){

		//--------------------------------------------------------------------------------------------//
		//--------------------------------------------------------------------------------------------//
		//--------------------------------------------------------------------------------------------//
		//------------------- INITIALIZATION OF VECTORS/ARRAYS THAT WILL BE USED ---------------------//
		//--------------------------------------------------------------------------------------------//
		//--------------------------------------------------------------------------------------------//
		//--------------------------------------------------------------------------------------------//
		// Storing AMR-selected particles
		std::vector<gridPoint>	Particle_Locations;
		std::vector<TYPE>		Particle_Values;

		// Full array storing appended particles for all parameter samples
		std::vector<gridPoint>	Full_Particle_Locations;
		std::vector<TYPE>		Full_Particle_Values;

		// Simulation logging
		Logger SimLog(time_vector.size() - 1);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 	PARAMETERS

		INT totalSampleCount = 1, sum_sampleNum = 0;
		INT Samples_per_Param[PARAM_SPACE_DIMENSIONS];

		for (UINT i = 0; i < PARAM_SPACE_DIMENSIONS; i++) {
			UINT temp = Parameter_Distributions[i].num_Samples;

			// Total samples from the parameter space discretization
			totalSampleCount 	 *= temp;
			
			// Length of the compressed parameter array
			sum_sampleNum 		 += temp;

			// Array storing the number of samples for each parameter
			Samples_per_Param[i]  = temp;
		}

		// Full parameter array
		Param_pair*	Parameter_Mesh = new Param_pair[sum_sampleNum];				

		// Build the parameter mesh from previous info
		int16_t error_check = RANDOMIZE<PARAM_SPACE_DIMENSIONS>(Parameter_Mesh, Parameter_Distributions);
		if (error_check == -1){return -1;}

		// Small vector containing the number of samples per parameter
		thrust::device_vector<INT>	D_sampVec(Samples_per_Param, Samples_per_Param + PARAM_SPACE_DIMENSIONS);	
		
		// Parameter Problem_Domain array (for the GPU)
		thrust::device_vector<Param_pair> D_Parameter_Mesh(Parameter_Mesh, Parameter_Mesh + sum_sampleNum);		

		// auxiliary variable that will be used for ensemble mean computation
		TYPE sum_sample_val = 0;
		for (UINT i = 0; i < totalSampleCount; i++) {
			Param_vec<PARAM_SPACE_DIMENSIONS> temp = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i, Parameter_Mesh, Samples_per_Param);
			sum_sample_val += temp.Joint_PDF;
		}

		// Output to the CLI
		std::cout << "Total number of random samples: " << totalSampleCount << ".\n";

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PROBLEM DOMAIN AND INITIAL PDF

		// This grid will be defined by the support bounding box of the data.
		grid PDF_Support(IC_InfTVAL, IC_SupTVAL);

		// Make it square for AMR-purposes
		PDF_Support.Squarify();

		// Update the number of pts per dimension
		PDF_Support.Nodes_per_Dim = round((IC_SupTVAL[0] - IC_InfTVAL[0]) / Problem_Domain.Discr_length());

		// PDF values at the fixed, high-res grid (CPU)
		thrust::host_vector<TYPE> PDF_ProbDomain(Problem_Domain.Total_Nodes(), 0);	 			

		// initialize the grid and the PDF at the grid nodes (change so as to change the parameters as well)
		error_check = PDF_INITIAL_CONDITION(Problem_Domain, PDF_ProbDomain, IC_Distributions); 	
		if (error_check == -1){return -1;}

		// PDF values at fixed Grid Nodes (for the GPU)
		thrust::device_vector<TYPE> D_PDF_ProbDomain = PDF_ProbDomain;

		// Particle positions (for the GPU)
		thrust::device_vector<gridPoint> D_Particle_Locations;
		
		// PDF value at Particle positions (for the GPU)
		thrust::device_vector<TYPE> D_Particle_Values;


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// max memory to be used (in bytes). 95% just in case
		const UINT MAX_BYTES_USEABLE = 0.95 * D_Properties->totalGlobalMem;		

		// ------------------ DEFINITION OF THE INTERPOLATION VARIABLES AND ARRAYS ------------------ //

		// The following consts don't need an explanation
		const UINT	ConjGrad_MaxSteps  = 1000;		 								
		const TYPE	RBF_SupportRadius  = DISC_RADIUS * Problem_Domain.Discr_length();

		// Now we make a slightly larger domain for the computations:
		grid Expanded_Domain;

		// Expand 40 nodes appears to be just fine
		const UINT expansion_nodes = 40;
		Expanded_Domain.Expand_From(Problem_Domain, expansion_nodes);

		// Simulation steps
		uint32_t simStepCount = 0;
		
		// Aux variable to switch between step functions (Heaviside forcing) 
		uint32_t mode = 0;	

		// IF THERE ARE DELTA TERMS
		#if IMPULSE_TYPE == 1
			UINT jumpCount = 0;	// auxiliary variable to know how many delta jumpCounts have passed
		#endif
		
		// IF THERE ARE HEAVISIDE TERMS WITH EXTRA PARAMETERS
		#if INCLUDE_XTRA_PARAMS
			thrust::device_vector<double>	Extra_Parameter(XTRA_PARAM_LENGTH);
			thrust::copy(&XTRA_PARAM[0], &XTRA_PARAM[XTRA_PARAM_LENGTH], Extra_Parameter.begin());
		#else
			thrust::device_vector<double> Extra_Parameter(0);
		#endif

		// The string to be used for printing to the console
		std::string printCLI;


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// START OF SIMULATION STEPS AS SUCH

		// Store the 1st PDF (Init. Cond.)
		thrust::copy(PDF_ProbDomain.begin(), PDF_ProbDomain.end(), &storeFrames[0]);

		// IN THIS LINE WE START WITH THE ACTUAL ITERATIONS OF THE LIOUVILLE EQUATION
		while (simStepCount < time_vector.size() - 1 && error_check != -1) {

			// select the first and last time value of the current iteration
			double	t0 = time_vector[simStepCount].time, tF = time_vector[simStepCount + 1].time;

			std::cout << "+---------------------------------------------------------------------+\n";

			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- ADAPT. MESH REFINEMENT --------------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			auto startTimeSeconds = std::chrono::high_resolution_clock::now();

				error_check = setInitialParticles(PDF_ProbDomain, D_PDF_ProbDomain,
													Particle_Values, Particle_Locations, 
													Problem_Domain, Expanded_Domain, PDF_Support);
				if (error_check == -1) { break; }

			auto endTimeSeconds = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> durationSeconds = endTimeSeconds - startTimeSeconds;

			// Enter the information into the log information
			SimLog.subFrame_time[5*simStepCount] = durationSeconds.count();
			
			// Number of particles to advect
			UINT AMR_ActiveNodeCount = Particle_Locations.size();

			// Clear the GPU-stored PDF for better memory availability
			D_PDF_ProbDomain.clear();

			// Send adapted values and points to the GPU
			D_Particle_Values.resize(AMR_ActiveNodeCount);
			D_Particle_Locations.resize(AMR_ActiveNodeCount);

			D_Particle_Values 	 = Particle_Values;
			D_Particle_Locations = Particle_Locations;

			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			// -------------------------- PT. SEARCH & INTERPOLATION ----------------------------- //
			/////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////
			uint16_t	Threads = fmin(THREADS_P_BLK, AMR_ActiveNodeCount);
			UINT		Blocks	= floor((AMR_ActiveNodeCount - 1) / Threads) + 1;

			// Maximum neighbors to search. Diameter number of points powered to the dimension
			UINT MaxNeighborNum = round(fmin(pow(2 * round(DISC_RADIUS) + 1, PHASE_SPACE_DIMENSIONS), AMR_ActiveNodeCount));
			
			// Compressed COO-style indexing of the sparse interpolation matrix
			thrust::device_vector<INT>	D_Mat_Indx(MaxNeighborNum * AMR_ActiveNodeCount, -1);

			// Sparse interpolation matrix values
			thrust::device_vector<TYPE>	D_Mat_Vals(MaxNeighborNum * AMR_ActiveNodeCount, 0);

			// Number of nonzeros per row (neighbors for each particle)
			thrust::device_vector<UINT>	D_Mat_NumNeighbors(AMR_ActiveNodeCount, 0);
			

			startTimeSeconds = std::chrono::high_resolution_clock::now();
				// Dynamical choice of either exhaustive or counting sort-based point search
				if (AMR_ActiveNodeCount < ptSEARCH_THRESHOLD) {
					Exh_PP_Search<< <Blocks, Threads >> > (rpc(D_Particle_Locations, 0),
														rpc(D_Particle_Locations, 0),
														rpc(D_Mat_Indx, 0),
														rpc(D_Mat_Vals, 0),
														rpc(D_Mat_NumNeighbors, 0),
														MaxNeighborNum,
														AMR_ActiveNodeCount,
														AMR_ActiveNodeCount,
														RBF_SupportRadius);
					gpuError_Check(cudaDeviceSynchronize());
				}
				else {
					error_check = CS_Neighbor_Search(D_Particle_Locations,
													D_Particle_Values,
													D_Mat_Indx,
													D_Mat_Vals,
													D_Mat_NumNeighbors,
													AMR_ActiveNodeCount,
													MaxNeighborNum,
													PDF_Support,
													RBF_SupportRadius);

					if (error_check == -1) { break; }
				}
			endTimeSeconds = std::chrono::high_resolution_clock::now();
			durationSeconds = endTimeSeconds - startTimeSeconds;
			
			// Enter the information into the log information
			SimLog.subFrame_time[5*simStepCount + 2] = durationSeconds.count();

			// Declare the solution of the interpolation vector (weights of the RBF functions)
			thrust::device_vector<TYPE> D_lambdas(AMR_ActiveNodeCount, 0);

			startTimeSeconds = std::chrono::high_resolution_clock::now();
				error_check = CONJUGATE_GRADIENT_SOLVE(	D_lambdas,
														D_Mat_Indx,
														D_Mat_Vals,
														D_Mat_NumNeighbors,
														D_Particle_Values,
														AMR_ActiveNodeCount,
														MaxNeighborNum,
														ConjGrad_MaxSteps,
														TOLERANCE_ConjGrad);
				if (error_check == -1) { std::cout << "Convergence failure.\n"; break; }
			endTimeSeconds = std::chrono::high_resolution_clock::now();
			durationSeconds = endTimeSeconds - startTimeSeconds;
			
			// Enter the information into the log information
			SimLog.subFrame_time[5*simStepCount + 3] = durationSeconds.count();
			SimLog.ConvergenceIterations[simStepCount] = error_check;

			// Clear the vectors to save memory
			D_Mat_Indx.clear();
			D_Mat_Vals.clear();
			D_Mat_NumNeighbors.clear();


			// FIX THIS PART!!
			if (time_vector[simStepCount].impulse) {

				#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-TYPE IMPULSE!

				std::cout << "RVT transformation at time: " << t0 << "\n";

				startTimeSeconds = std::chrono::high_resolution_clock::now();

				error_check = IMPULSE_TRANSFORM_PDF<PHASE_SPACE_DIMENSIONS, TYPE>(D_PDF_ProbDomain,
																		Particle_Locations,
																		Particle_Values,
																		time_vector[simStepCount],
																		jumpCount,
																		Problem_Domain,
																		Expanded_Domain,
																		PDF_Support);

				endTimeSeconds = std::chrono::high_resolution_clock::now();
				durationSeconds = endTimeSeconds - startTimeSeconds;

				
				// Enter the information into the log information
				SimLog.subFrame_time[5*simStepCount + 1] = durationSeconds.count();

				assert(error_check == 0);

				Particle_Locations.clear();
				Particle_Values.clear();

				jumpCount++;
				simStepCount++;

				PDF_ProbDomain = D_PDF_ProbDomain;

				// Store info in cumulative variable
				thrust::copy(PDF_ProbDomain.begin(), PDF_ProbDomain.end(), &(*storeFrames)[simStepCount * Problem_Domain.Total_Nodes()]);

				#elif(IMPULSE_TYPE == 2)	// THIS IS FOR HEAVISIDE-TYPE IMPULSE!
				mode++;
				std::cout << "Now the vector field is in mode: " << mode % 2 << ".\n";

				Particle_Locations.clear();
				Particle_Values.clear();

				simStepCount++;

				#elif(IMPULSE_TYPE != 0)
				std::cout << "Error in 'Dynamics.cuh'. You are choosing an unavailable option. Go back to 'Case_definition.cuh' and re-check options for IMPULSE_TYPE.\n";
				error_check = -1;
				break;

				#endif
			}
			// 1.2.- COMPUTE THE SMOOTH EVOLUTION VIA LIOUVILLE GIBBS/CONTINUITY EQUATION
			else {
				std::cout << "Simulation time: " << t0 << " to " << tF << "\n";

				// Max. memory requirements for next Liouville step
				const UINT Bytes_per_sample = AMR_ActiveNodeCount * (sizeof(TYPE) * 2 + sizeof(gridPoint));

				// Set number of random samples to work with at the same time
				UINT Samples_PerBlk = fmin((UINT)totalSampleCount, MAX_BYTES_USEABLE / Bytes_per_sample);
				
				// Number of blocks to simulate
				UINT total_simulation_blocks = (UINT) ceilf((float)totalSampleCount / Samples_PerBlk);

				for (UINT b = 0; b < total_simulation_blocks; b++) {

					// Parameter sample offset init. and final to account for the block position
					UINT Sample_idx_offset_init = b * Samples_PerBlk;
					UINT Sample_idx_offset_final = fmin((b + 1) * Samples_PerBlk, totalSampleCount);

					// Actual number of samples in current block
					Samples_PerBlk = Sample_idx_offset_final - Sample_idx_offset_init;

					// Total AMR-activated nodes in the current block
					UINT ActiveNodes_PerBlk = Samples_PerBlk * AMR_ActiveNodeCount;

					// 1.2.- Append the optimal particles once per sample!
					Full_Particle_Locations.resize(ActiveNodes_PerBlk);
					Full_Particle_Values   .resize(ActiveNodes_PerBlk);

					for (UINT k = 0; k < Samples_PerBlk; k++) {
						std::copy(Particle_Locations.begin(), Particle_Locations.end(), &Full_Particle_Locations[k * AMR_ActiveNodeCount]);
						std::copy(Particle_Values.begin(), Particle_Values.end(), &Full_Particle_Values[k * AMR_ActiveNodeCount]);
					}

					// Upload information to the GPU
					D_Particle_Values.resize(Samples_PerBlk);
					D_Particle_Locations.resize(Samples_PerBlk);

					D_Particle_Values 	 = Full_Particle_Values;
					D_Particle_Locations = Full_Particle_Locations;

					// Clear CPU info about the AMR procedure: no longer needed
					Full_Particle_Locations.clear();
					Full_Particle_Values.clear();

					// Print the active nodes in the current step
					printCLI = "Number of active particles: " + std::to_string(AMR_ActiveNodeCount);
					std::cout << printCLI << "\n";


					/////////////////////////////////////////////////////////////////////////////////////////
					/////////////////////////////////////////////////////////////////////////////////////////
					// -------------------------- POINT ADVECTION ---------------------------------------- //
					/////////////////////////////////////////////////////////////////////////////////////////
					/////////////////////////////////////////////////////////////////////////////////////////
					Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
					Blocks	= floor((ActiveNodes_PerBlk - 1) / Threads) + 1;

					startTimeSeconds = std::chrono::high_resolution_clock::now();
						ODE_INTEGRATE<< <Blocks, Threads >> > (rpc(D_Particle_Locations, 0),
																rpc(D_Particle_Values, 0),
																rpc(D_lambdas, 0),
																rpc(D_Parameter_Mesh, Sample_idx_offset_init),
																rpc(D_sampVec, 0),
																t0,
																deltaT,
																tF,
																AMR_ActiveNodeCount,
																Samples_PerBlk,
																mode,
																rpc(Extra_Parameter, 0),
																Problem_Domain);
						gpuError_Check(cudaDeviceSynchronize());
					endTimeSeconds = std::chrono::high_resolution_clock::now();
					durationSeconds = endTimeSeconds - startTimeSeconds;

					// To the Log
					SimLog.subFrame_time[5*simStepCount + 1] = durationSeconds.count();

					PDF_Support.Update_boundingBox(D_Particle_Locations);
					
					// ----------------------------------------------------------------------------------- //
					// COMPUTE THE SOLUTION "PROJECTION" INTO THE L1 SUBSPACE
					if (PHASE_SPACE_DIMENSIONS < 3) {
						TYPE temp = thrust::reduce(thrust::device, D_Particle_Values.begin(), D_Particle_Values.end());
						thrust::transform(D_Particle_Values.begin(), D_Particle_Values.end(), D_Particle_Values.begin(), Samples_PerBlk / temp * _1);
					}
					// ----------------------------------------------------------------------------------- //

					/////////////////////////////////////////////////////////////////////////////////////////
					/////////////////////////////////////////////////////////////////////////////////////////
					// -------------------------- REINITIALIZATION --------------------------------------- //
					/////////////////////////////////////////////////////////////////////////////////////////
					/////////////////////////////////////////////////////////////////////////////////////////

					D_PDF_ProbDomain.resize(Problem_Domain.Total_Nodes(), 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
					
					Threads = fminf(THREADS_P_BLK, ActiveNodes_PerBlk);
					Blocks  = floorf((ActiveNodes_PerBlk - 1) / Threads) + 1;

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
																	Problem_Domain,
																	Expanded_Domain);
					gpuError_Check(cudaDeviceSynchronize());
					endTimeSeconds = std::chrono::high_resolution_clock::now();
					durationSeconds = endTimeSeconds - startTimeSeconds;

					
					// Enter the information into the log information
					SimLog.subFrame_time[5 * simStepCount + 4] = durationSeconds.count();
				}

				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////
				// -------------------------- STORE PDF INTO OUTPUT ARRAY ---------------------------- //
				/////////////////////////////////////////////////////////////////////////////////////////
				/////////////////////////////////////////////////////////////////////////////////////////

				// Clean all particle-related variables before continuing to the next time instant 
				Particle_Locations.clear();
				Particle_Values.clear();

				// Correction of any possible negative PDF values
				UINT Threads = fminf(THREADS_P_BLK, Problem_Domain.Total_Nodes());
				UINT Blocks = floorf((Problem_Domain.Total_Nodes() - 1) / Threads) + 1;

				CORRECTION<< <Blocks, Threads >> > (rpc(D_PDF_ProbDomain, 0), Problem_Domain.Total_Nodes());
				gpuError_Check(cudaDeviceSynchronize());

				// Divide by the sum of the values of the parameter mesh to obtain the weighted mean
				thrust::transform(D_PDF_ProbDomain.begin(), D_PDF_ProbDomain.end(), D_PDF_ProbDomain.begin(), 1.0f / sum_sample_val * _1); // we use the thrust::placeholders here (@ the last input argument)
	
				// Send back to CPU
				PDF_ProbDomain = D_PDF_ProbDomain;

				// Write some log info to the command line
				SimLog.writeToCLI(OUTPUT_INFO, simStepCount);

				std::cout << "+---------------------------------------------------------------------+\n";

				// Upadte simulation step
				simStepCount++;

				// Store info in cumulative variable
				thrust::copy(PDF_ProbDomain.begin(), PDF_ProbDomain.end(), &storeFrames[simStepCount * Problem_Domain.Total_Nodes()]);
			}
		}

		// Write entire log to a log file!
		SimLog.writeToFile();
		
		// Memory management
		delete[] Parameter_Mesh;

		// Exit current function
		return error_check;

	}

	int16_t WriteFramesToFile(const double& simulationDuration){
            // We have added the capability of automatically detecting the number of 1 GB files where we can store the simulation output
            bool saving_active = true;		// see if saving is still active
            int16_t error_check = 0;

            const uint64_t MEM_2_STORE		= storeFrames.size() * sizeof(float);
            
            UINT number_of_frames_needed 	= MEM_2_STORE / Problem_Domain.Total_Nodes() / sizeof(float);
            uint64_t max_frames_file 		= (uint64_t)MAX_FILE_SIZE_B / Problem_Domain.Total_Nodes() / sizeof(float);
            UINT number_of_files_needed  	= floor((number_of_frames_needed - 1) / max_frames_file) + 1;
            
            char ans;
            std::cout << "Simulation time: " << simulationDuration << " seconds. ";
            
            if(number_of_files_needed == 0){
                std::cout << "There has been a problem. No memory written. Exiting simulation.\n";
                saving_active = false;
                error_check = -1;
            }

            while(saving_active){	
                std::cout << "Total memory of simulation: " << (float) MEM_2_STORE / 1024/1024 << " MB. \n";
                std::cout << number_of_files_needed << " files required for total storage. Total frames: " << number_of_frames_needed << ", with frames per file: " << max_frames_file << " \n";
                std::cout << "Write? (Y = Yes(total), N = No, P = Partial): ";
                std::cin >> ans;

                while((ans != 'N') && (ans != 'n') && (ans != 'Y') && (ans != 'y') && (ans != 'P') && (ans != 'p')){
                    std::cout << "Incorrect option. Choose one of the following (NOT case sensitive: Y = Yes, N = No, P = Partial): ";
                    std::cin >> ans;
                }


                if ((ans != 'N') && (ans != 'n')) {

                    INT frames_init = 0, frames_end = number_of_files_needed - 1;
                    bool condition = false;

                    if((ans == 'P') || (ans == 'p')){
                        while(!condition){
                            std::cout << "Initial frame (must be >= 0): ";
                            std::cin >> frames_init;
                            std::cout << "Final frame (must be < "<< number_of_frames_needed <<"): ";
                            std::cin >> frames_end;

                            if(frames_init < 0 || frames_end >= number_of_frames_needed || frames_init > frames_end){

                                if(frames_init == -1 || frames_end == -1){
                                    std::cout << "Exiting simulation without saving simulation results...\n";
                                    return -1;
                                }
                                std::cout << "Check numbers, something's not right...\n";
                            }
                            else{
                                condition = true;
                                number_of_frames_needed = frames_end - frames_init + 1;
                                number_of_files_needed  = floor((number_of_frames_needed - 1) / max_frames_file) + 1;
                            }
                        }
                    }
                    else{
                        frames_init = 0, frames_end = number_of_files_needed - 1;
                    }

                    #pragma omp parallel for
                    for(int16_t k = 0; k < number_of_files_needed; k++){

                        UINT frames_in_file = fmin(max_frames_file, number_of_frames_needed - k * max_frames_file);

                        std::string temp_str = std::to_string((UINT)k);

                    	// SIMULATION INFORMATION FILE
                        std::string relavtive_pth = SIM_OUTPUT_relPATH;
                        relavtive_pth.append("Simulation_info_");
                        relavtive_pth.append(temp_str);
                        relavtive_pth.append(".csv");

                        std::ofstream file1(relavtive_pth, std::ios::out);
                        assert(file1.is_open());

                        file1 << Problem_Domain.Total_Nodes() << "," << Problem_Domain.Nodes_per_Dim << ",";

                        for (UINT d = 0; d < PHASE_SPACE_DIMENSIONS; d++){
                                  file1 << Problem_Domain.Boundary_inf.dim[d] << "," << Problem_Domain.Boundary_sup.dim[d] << ",";
                        }
                        for (uint16_t d = 0; d < PARAM_SPACE_DIMENSIONS; d++) {
                            file1 << Parameter_Distributions[d].num_Samples << ",";
                        }
                        file1 << simulationDuration << "\n";

                        #if IMPULSE_TYPE == 0 || IMPULSE_TYPE ==1
                        for (UINT i = k * max_frames_file + frames_init; i < k * max_frames_file + frames_in_file + frames_init - 1; i++) {
                            file1 << time_vector[i].time << ",";
                        }
                        file1 << time_vector[k * max_frames_file + frames_in_file + frames_init - 1].time;

                        #elif IMPULSE_TYPE == 2
                        file1 << time_vector[k * max_frames_file + frames_init].time << ",";

                        for (UINT i = k * max_frames_file + 1 + frames_init; i < k * max_frames_file + frames_init + frames_in_file; i++) {
                            if (abs(time_vector[i].time - time_vector[i - 1].time) > pow(10, -7)) {
                                file1 << time_vector[i].time << ",";
                            }
                            else if (i == k * max_frames_file + frames_in_file + frames_init - 1) {
                                if (time_vector[i].time != time_vector[i - 1].time) {
                                    file1 << time_vector[i].time;
                                }
                            }
                        }
                        #endif
                        file1.close();

                    	// SIMULATION OUTPUT
                        relavtive_pth = SIM_OUTPUT_relPATH;
                        relavtive_pth.append("Mean_PDFs_");
                        relavtive_pth.append(temp_str);
                        relavtive_pth.append(".bin");

                        std::ofstream myfile(relavtive_pth, std::ios::out | std::ios::binary);
                        assert (myfile.is_open());

                        myfile.write((char*)&storeFrames[(k * max_frames_file + frames_init) * Problem_Domain.Total_Nodes()], sizeof(float) * frames_in_file * Problem_Domain.Total_Nodes());
                        myfile.close();
                        std::cout << "Simulation output file " << k << " completed!\n";
                        
					}
				}
				saving_active = false;
			}
			return error_check;
		}
};
}

#endif