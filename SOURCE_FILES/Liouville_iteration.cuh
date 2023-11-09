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

#ifndef __LIOUVILLE_ITERATION_CUH__
#define __LIOUVILLE_ITERATION_CUH__

#include "Simulation_parameters.cuh"
#include "Parameter_distributions.cuh"
#include "Adapt_Mesh.cuh"
#include "Interpolation.cuh"
#include "Impulse_transformations.cuh"
#include "Integrator.cuh"

#include "Debugging.cuh"

/// @brief
/// @tparam T 
/// @tparam SPACE_DIM 
/// @tparam PARAM_DIM 
/// @param D_Properties 
/// @param store_PDFs 
/// @param Parameter_Distribution_Info 
/// @param IC_Distribution_Info 
/// @param time_vector 
/// @param deltaT 
/// @param ReinitSteps 
/// @return 
template<uint16_t SPACE_DIM, uint16_t PARAM_DIM, class T>
int16_t PDF_ITERATIONS(	cudaDeviceProp*			D_Properties,
						std::vector<T>*			store_PDFs,
						const grid<SPACE_DIM,T>&	ProblemDomain_Mesh,
						const Distributions*	Parameter_Distribution_Info,
						const Distributions*	IC_Distribution_Info,
						const std::vector<Time_Impulse_vec> time_vector,
						const double&			deltaT,
						const UINT&				ReinitSteps) {

	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	//------------------- INITIALIZATION OF VECTORS/ARRAYS THAT WILL BE USED ---------------------//
	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	std::vector<gridPoint<SPACE_DIM, T>>	Particle_Locations;		// Particle positions to be used for simulation (corresponding dim from AMR)
	std::vector<T>							Particle_Values;		// PDF value at the particle positions (corresponding values from AMR)
	std::vector<gridPoint<SPACE_DIM, T>>	Full_Particle_Locations;// Final adapted grid<SPACE_DIM, T> (adapted grid<SPACE_DIM, T> x number of samples)
	std::vector<T>							Full_Particle_Values;	// Final adapted PDF (adapted grid<SPACE_DIM, T> x number of samples)

	Logger SimLog(time_vector.size() - 1);							// Simulation logging

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 	PARAMETERS
	INT total_samples = 1, sum_sampleNum = 0;
	INT Samples_per_Param[PARAM_DIM];

	for (UINT i = 0; i < PARAM_DIM; i++) {
		UINT temp = Parameter_Distribution_Info[i].num_Samples;

		total_samples 	*= temp;
		sum_sampleNum 	+= temp;
		Samples_per_Param[i] 		=  temp;
	}

	// Full parameter array
	Param_pair*	Parameter_Mesh = new Param_pair[sum_sampleNum];				

	// CALL JOINT PDF BUILDING FUNCTION: (PARAMETERS AND INITIAL CONDITION)
	int16_t error_check = RANDOMIZE(Parameter_Mesh, Parameter_Distribution_Info);
	if (error_check == -1){return -1;}

	// GPU arrays //
	// Small vector containing the number of samples per parameter
	thrust::device_vector<INT>	D_sampVec(Samples_per_Param, Samples_per_Param + PARAM_DIM);	
	std::cout << "Total number of random samples: " << total_samples << ".\n";
	
	// Parameter ProblemDomain_Mesh array (for the GPU)
	thrust::device_vector<Param_pair> D_Parameter_Mesh(Parameter_Mesh, Parameter_Mesh + sum_sampleNum);		

	// auxiliary variable that will be used for ensemble mean computation
	T sum_sample_val = 0;
	for (UINT i = 0; i < total_samples; i++) {
		Param_vec temp = Gather_Param_Vec(i, Parameter_Mesh, Samples_per_Param);
		sum_sample_val += temp.Joint_PDF;
	}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DOMAIN STUFF
	// This one will be defined by the support bounding box of the data. Now we need to get the number of points needed! And, we have to "squarify" it
	grid<DIMENSIONS, TYPE> PDFsupport_Mesh(IC_InfTVAL, IC_SupTVAL);
	PDFsupport_Mesh.Squarify();
	PDFsupport_Mesh.Nodes_per_Dim = round((IC_SupTVAL[0] - IC_InfTVAL[0]) / ProblemDomain_Mesh.Discr_length());

	thrust::host_vector<TYPE> PDF_at_ProbDom(ProblemDomain_Mesh.Total_Nodes(), 0);	 			// PDF values at the fixed, high-res grid (CPU)

	// initialize the grid and the PDF at the grid nodes (change so as to change the parameters as well)
	error_check = PDF_INITIAL_CONDITION(ProblemDomain_Mesh, PDF_at_ProbDom, IC_Distribution_Info); 	
	if (error_check == -1){return -1;}

	thrust::device_vector<T>					D_PDF_at_ProbDom = PDF_at_ProbDom;		// PDF values at fixed Grid Nodes (for the GPU)

	thrust::device_vector<gridPoint<SPACE_DIM, T>>	D_Particle_Locations;	// Particle positions (for the GPU)
	thrust::device_vector<T>						D_Particle_Values;			// PDF value at Particle positions (for the GPU)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// max memory to be used (in bytes). 95% just in case
	const UINT MAX_MEMORY_USABLE = 0.95 * D_Properties->totalGlobalMem;		

	// ------------------ DEFINITION OF THE INTERPOLATION VARIABLES AND ARRAYS ------------------ //
	UINT AMR_ActiveNodes, MaxNeighborNum;

	const UINT	ConjGrad_MaxSteps  = 1000;		 								
	const T 	ConjGrad_Tolerance = TOLERANCE_ConjGrad, 
				RBF_SupportRadius  = DISC_RADIUS * ProblemDomain_Mesh.Discr_length();

	// Now we make a slightly larger domain for the computations:
	grid<DIMENSIONS, TYPE> Underlying_Mesh;

	const UINT expansion_nodes = 40;
	Underlying_Mesh.Expand_From(ProblemDomain_Mesh, expansion_nodes);

	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// ---------------------------------- SIMULATION TIME!! ---------------------------------------
	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------

	// -------------------- Store the 1st PDF (Init. Cond.) -----------------------------------
	thrust::copy(PDF_at_ProbDom.begin(), PDF_at_ProbDom.end(), &(*store_PDFs)[0]);

	// ------------------------------------------------------------------------------------
	uint16_t simStepCount = 0, mode = 0;

	// IF THERE ARE DELTA TERMS
	#if IMPULSE_TYPE == 1
		UINT jump = 0;	// auxiliary variable to know how many delta jumps have passed
	#endif
	
	// IF THERE ARE HEAVISIDE TERMS WITH EXTRA PARAMETERS
	#if INCLUDE_XTRA_PARAMS
		thrust::device_vector<double>	Extra_Parameter(XTRA_PARAM_LENGTH);
		thrust::copy(&XTRA_PARAM[0], &XTRA_PARAM[XTRA_PARAM_LENGTH], Extra_Parameter.begin());
	#else
		thrust::device_vector<double>	Extra_Parameter(0);
	#endif


// IN THIS LINE WE COMMENCE WITH THE ACTUAL ITERATIONS OF THE LIOUVILLE EQUATION
	while (simStepCount < time_vector.size() - 1 && error_check != -1) {

		double	t0 = time_vector[simStepCount].time, tF = time_vector[simStepCount + 1].time;

		std::cout << "+---------------------------------------------------------------------+\n";
		// 1.- Initial step Adaptive ProblemDomain_Mesh Refinement. First store the initial PDF with AMR performed

	auto startTimeSeconds = std::chrono::high_resolution_clock::now();

		error_check = ADAPT_MESH_REFINEMENT_nD<SPACE_DIM, T>(PDF_at_ProbDom, D_PDF_at_ProbDom,
														 Particle_Values, Particle_Locations, 
														ProblemDomain_Mesh, Underlying_Mesh, PDFsupport_Mesh);
		if (error_check == -1) { break; }

	auto endTimeSeconds = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration_3 = endTimeSeconds - startTimeSeconds;

		// Enter the information into the log information
		SimLog.subFrame_time[5*simStepCount] = duration_3.count();
		
		// Number of particles to advect
		AMR_ActiveNodes = Particle_Locations.size();

		D_PDF_at_ProbDom.clear();

		// Send adapted values and points to the GPU
		D_Particle_Values.resize(AMR_ActiveNodes);
		D_Particle_Locations.resize(AMR_ActiveNodes);

		D_Particle_Values 	= Particle_Values;
		D_Particle_Locations 	= Particle_Locations;

		// ----------------------------------------------------------------------------------- //
		// -------------------------- INTERPOLATION ------------------------------------------ //
		// ----------------------------------------------------------------------------------- //
		
		uint16_t	Threads = fmin(THREADS_P_BLK, AMR_ActiveNodes);
		UINT		Blocks	= floor((AMR_ActiveNodes - 1) / Threads) + 1;

		// Maximum neighbors to search. Diameter number of points powered to the dimension
		MaxNeighborNum = round(fmin(pow(2 * round(DISC_RADIUS) + 1, SPACE_DIM), AMR_ActiveNodes));

		// ------------------ RESIZING OF THE INTERPOLATION MATRIX ------------------ //
			thrust::device_vector<INT>	D_Mat_Indx(MaxNeighborNum * AMR_ActiveNodes, -1);
			thrust::device_vector<T>	D_Mat_Vals(MaxNeighborNum * AMR_ActiveNodes, 0);
			thrust::device_vector<UINT>	D_Mat_NumNeighbors(AMR_ActiveNodes, 0);
		// -------------------------------------------------------------------------- //
		// 1.- Build Matix in GPU (indexes, dists and neighbors) Using Exahustive search...
		startTimeSeconds = std::chrono::high_resolution_clock::now();

		// Dynamical choice of either exhaustive or counting sort-based point search
		if (AMR_ActiveNodes < ptSEARCH_THRESHOLD) {
			Exh_PP_Search<SPACE_DIM,T> << <Blocks, Threads >> > (rpc(D_Particle_Locations, 0),
																rpc(D_Particle_Locations, 0),
																rpc(D_Mat_Indx, 0),
																rpc(D_Mat_Vals, 0),
																rpc(D_Mat_NumNeighbors, 0),
																MaxNeighborNum,
																AMR_ActiveNodes,
																AMR_ActiveNodes,
																RBF_SupportRadius);
			gpuError_Check(cudaDeviceSynchronize());
		}
		else {
			error_check = _CS_Neighbor_Search<SPACE_DIM, T>(D_Particle_Locations,
															D_Particle_Values,
															D_Mat_Indx,
															D_Mat_Vals,
															D_Mat_NumNeighbors,
															AMR_ActiveNodes,
															MaxNeighborNum,
															PDFsupport_Mesh,
															RBF_SupportRadius);

			if (error_check == -1) { break; }
		}

		endTimeSeconds = std::chrono::high_resolution_clock::now();
		duration_3 = endTimeSeconds - startTimeSeconds;
		
		// Enter the information into the log information
		SimLog.subFrame_time[5*simStepCount + 2] = duration_3.count();


		// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
		thrust::device_vector<T> D_lambdas(AMR_ActiveNodes);		// solution vector (RBF weights)
		thrust::fill(D_lambdas.begin(), D_lambdas.end(), 0.0f);	// this will serve as the initial condition

		startTimeSeconds = std::chrono::high_resolution_clock::now();
		error_check = CONJUGATE_GRADIENT_SOLVE<T>(	D_lambdas,
													D_Mat_Indx,
													D_Mat_Vals,
													D_Mat_NumNeighbors,
													D_Particle_Values,
													AMR_ActiveNodes,
													MaxNeighborNum,
													ConjGrad_MaxSteps,
													ConjGrad_Tolerance);
		if (error_check == -1) { std::cout << "Convergence failure.\n"; break; }
		endTimeSeconds = std::chrono::high_resolution_clock::now();
		duration_3 = endTimeSeconds - startTimeSeconds;
		
		// Enter the information into the log information
		SimLog.subFrame_time[5*simStepCount + 3] = duration_3.count();
		SimLog.ConvergenceIterations[simStepCount] = error_check;

		// Clear them, to save memory
		D_Mat_Indx.clear();
		D_Mat_Vals.clear();
		D_Mat_NumNeighbors.clear();


		// 1.1.- COMPUTE THE TRANSFORMATION OF THE PDF (IF THERE IS ONE)
		if (time_vector[simStepCount].impulse) {

			#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-T IMPULSE!

			std::cout << "RVT transformation at time: " << t0 << "\n";

			startTimeSeconds = std::chrono::high_resolution_clock::now();

			error_check = IMPULSE_TRANSFORM_PDF<SPACE_DIM, T>(Particle_Locations,
																PDF_at_ProbDom,
																D_PDF_at_ProbDom,
																Particle_Values,
																time_vector[simStepCount],
																jump,
																ProblemDomain_Mesh,
																Underlying_Mesh,
																PDFsupport_Mesh);

			endTimeSeconds = std::chrono::high_resolution_clock::now();
			duration_3 = endTimeSeconds - startTimeSeconds;

			
			// Enter the information into the log information
			SimLog.subFrame_time[5*simStepCount + 1] = duration_3.count();

			assert(error_check == 0);

			Particle_Locations.clear();
			Particle_Values.clear();

			jump++;
			simStepCount++;

			// Store info in cumulative variable
			thrust::copy(PDF_at_ProbDom.begin(), PDF_at_ProbDom.end(), &(*store_PDFs)[simStepCount * ProblemDomain_Mesh.Total_Nodes()]);

			#elif(IMPULSE_TYPE == 2)	// THIS IS FOR HEAVISIDE-T IMPULSE!
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
			const UINT mem_requested_per_sample = (UINT)AMR_ActiveNodes * (sizeof(T) * 2 + sizeof(gridPoint<SPACE_DIM, T>));

			// Set number of random samples to work with, and number of blocks to use
			UINT Samples_PerBlk = (UINT)fmin((UINT)total_samples, MAX_MEMORY_USABLE / mem_requested_per_sample);
			UINT total_simulation_blocks = (UINT)ceilf((float)total_samples / Samples_PerBlk);

			for (UINT b = 0; b < total_simulation_blocks; b++) {

				// Parameter sample offset to account for the block position
				UINT Sample_idx_offset_init = b * Samples_PerBlk;
				UINT Sample_idx_offset_final = fmin((b + 1) * Samples_PerBlk, total_samples);

				// Actual number of samples in current block
				Samples_PerBlk = Sample_idx_offset_final - Sample_idx_offset_init;

				UINT ActiveNodes_PerBlk = Samples_PerBlk * AMR_ActiveNodes;

				// 1.2.- Append the optimal particles once per sample!
				Full_Particle_Locations.resize(ActiveNodes_PerBlk);
				Full_Particle_Values   .resize(ActiveNodes_PerBlk);

				for (UINT k = 0; k < Samples_PerBlk; k++) {
					std::copy(Particle_Locations.begin(), Particle_Locations.end(), &Full_Particle_Locations[k * AMR_ActiveNodes]);
					std::copy(Particle_Values.begin(), Particle_Values.end(), &Full_Particle_Values[k * AMR_ActiveNodes]);
				}

				// Upload information to the GPU
				D_Particle_Values.resize(Samples_PerBlk);
				D_Particle_Locations.resize(Samples_PerBlk);

				D_Particle_Values 	= Full_Particle_Values;
				D_Particle_Locations 	= Full_Particle_Locations;

				// Clear CPU info about the AMR procedure: no longer needed
				Full_Particle_Locations.clear();
				Full_Particle_Values.clear();

				std::cout << "Size of relevant PDF points (per sample): " << AMR_ActiveNodes << "\n";	// this allows to check if the info is passed to the GPU correctly

				// Determine threads and blocks for the simulation
				Threads = fmin(THREADS_P_BLK, ActiveNodes_PerBlk);
				Blocks	= floor((ActiveNodes_PerBlk - 1) / Threads) + 1;

				// ------------------------------------------------------------------------------------ //
				// -------------------------- POINT ADVECTION ----------------------------------------- //
				// ------------------------------------------------------------------------------------ //
				startTimeSeconds = std::chrono::high_resolution_clock::now();
				ODE_INTEGRATE<SPACE_DIM, T><< <Blocks, Threads >> > (rpc(D_Particle_Locations, 0),
																	rpc(D_Particle_Values, 0),
																	rpc(D_lambdas, 0),
																	rpc(D_Parameter_Mesh, Sample_idx_offset_init),
																	rpc(D_sampVec, 0),
																	t0,
																	deltaT,
																	tF,
																	AMR_ActiveNodes,
																	Samples_PerBlk,
																	mode,
																	rpc(Extra_Parameter, 0),
																	ProblemDomain_Mesh);
				gpuError_Check(cudaDeviceSynchronize()); // Here, the entire ProblemDomain_Mesh points (those that were selected) and PDF points (same) have been updated.

				endTimeSeconds = std::chrono::high_resolution_clock::now();
				duration_3 = endTimeSeconds - startTimeSeconds;

				// Enter the information into the log information
				SimLog.subFrame_time[5*simStepCount + 1] = duration_3.count();

				// Before going to the next step, define the bounding box of the advected particles!
					thrust::device_vector<T> projection(ActiveNodes_PerBlk,(T)0);

					for (uint16_t d = 0; d < SPACE_DIM; d++) {
						findProjection<SPACE_DIM, T> << <Blocks, Threads >> > (rpc(D_Particle_Locations, 0), rpc(projection, 0), ActiveNodes_PerBlk, d);

						T temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end())); // min element from the projection in that direction
						T temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));


						// Eliminate the need for it to be in the Problem domain
						PDFsupport_Mesh.Boundary_inf.dim[d] = temp_1 - ceil(DISC_RADIUS) * ProblemDomain_Mesh.Discr_length();
						PDFsupport_Mesh.Boundary_sup.dim[d] = temp_2 + ceil(DISC_RADIUS) * ProblemDomain_Mesh.Discr_length();
					}
				projection.clear();

			
			// ----------------------------------------------------------------------------------- //
			// COMPUTE ITS "PROJECTION" INTO THE L1 SUBSPACE			
			if (SPACE_DIM < 3) {
				T temp = thrust::reduce(thrust::device, D_Particle_Values.begin(), D_Particle_Values.end());
				thrust::transform(D_Particle_Values.begin(), D_Particle_Values.end(), D_Particle_Values.begin(), Samples_PerBlk / temp * _1);
			}
			// ----------------------------------------------------------------------------------- //

			// 3.- Multiplication of matrix-lambdas to obtain updated grid<SPACE_DIM, T> nodes
				// I'M GOING TO FIND THE NEAREST GRID NODES TO EACH PARTICLE

				D_PDF_at_ProbDom.resize(ProblemDomain_Mesh.Total_Nodes(), 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
				
				Threads = fminf(THREADS_P_BLK, ActiveNodes_PerBlk);
				Blocks  = floorf((ActiveNodes_PerBlk - 1) / Threads) + 1;

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				RESTART_GRID_FIND_GN<SPACE_DIM,T> << < Blocks, Threads >> > (	rpc(D_Particle_Locations, 0),
																				rpc(D_PDF_at_ProbDom, 0),
																				rpc(D_Particle_Values, 0),
																				rpc(D_Parameter_Mesh, 0),
																				rpc(D_sampVec, 0),
																				RBF_SupportRadius,
																				AMR_ActiveNodes,
																				Samples_PerBlk,
																				Sample_idx_offset_init,
																				ProblemDomain_Mesh,
																				Underlying_Mesh);
				gpuError_Check(cudaDeviceSynchronize());
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				duration_3 = endTimeSeconds - startTimeSeconds;

				
				// Enter the information into the log information
				SimLog.subFrame_time[5 * simStepCount + 4] = duration_3.count();
			}

			Particle_Locations.clear();
			Particle_Values.clear();
			// Correction of any possible negative PDF values
			// Re-define Threads and Blocks
			UINT Threads = fminf(THREADS_P_BLK, ProblemDomain_Mesh.Total_Nodes());
			UINT Blocks = floorf((ProblemDomain_Mesh.Total_Nodes() - 1) / Threads) + 1;

			CORRECTION<T> << <Blocks, Threads >> > (rpc(D_PDF_at_ProbDom, 0), ProblemDomain_Mesh.Total_Nodes());
			gpuError_Check(cudaDeviceSynchronize());

			thrust::transform(D_PDF_at_ProbDom.begin(), D_PDF_at_ProbDom.end(), D_PDF_at_ProbDom.begin(), 1.0f / sum_sample_val * _1); // we use the thrust::placeholders here (@ the last input argument)

			PDF_at_ProbDom = D_PDF_at_ProbDom; // Send back to CPU

			SimLog.writeToCLI(OUTPUT_INFO, simStepCount);

			std::cout << "+---------------------------------------------------------------------+\n";

			simStepCount++;

			// Store info in cumulative variable
			thrust::copy(PDF_at_ProbDom.begin(), PDF_at_ProbDom.end(), &(*store_PDFs)[simStepCount * ProblemDomain_Mesh.Total_Nodes()]);
		}
	}

	SimLog.writeToFile();
	
	delete[] Parameter_Mesh;

	return error_check;
}
#endif