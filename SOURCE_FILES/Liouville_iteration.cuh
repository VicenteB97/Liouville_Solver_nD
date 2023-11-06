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

/// @brief This function computes the Liouville Eq. iterations from t0 to tF. Each iteration consists of the following steps:
/// 1st) Compute the AMR of the initial PDF. 
/// 2nd) Create as many particle families as random samples there are.
/// 3rd) Advect all particles via RungeKutta function. 
/// 4th) Interpolate all families of particles onto the underlying high-res. fixed grid<DIM, T>. 
/// 5th) Compute ensemble mean.
/// 6th) Store final PDF into the iteration vector for further post-processing and/or evolution visualization.
/// @param store_PDFs 
/// @param Parameter_Mesh 
/// @param Problem_Domain 
/// @param H_PDF 
/// @param PtsPerDim 
/// @param Problem_Domain
/// @param time_vector 
/// @param deltaT 
/// @param ReinitSteps 
/// @return 
template<uint16_t DIM, class T>
int16_t PDF_ITERATIONS(	cudaDeviceProp*			prop,
						std::vector<float>*		store_PDFs,
						const Param_pair*		Parameter_Mesh,
						const grid<DIM, T>&		Problem_Domain,
						grid<DIM, T>&			Supp_BBox,
						thrust::host_vector<T>*	H_PDF,
						const INT*				n_Samples,
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
	std::vector<gridPoint<DIM, T>>	AdaptGrid;			// Particle positions to be used for simulation (corresponding dim from AMR)
	std::vector<T>					AdaptPDF;			// PDF value at the particle positions (corresponding values from AMR)
	std::vector<gridPoint<DIM, T>>	Full_AdaptGrid;		// Final adapted grid<DIM, T> (adapted grid<DIM, T> x number of samples)
	std::vector<T>					Full_AdaptPDF;		// Final adapted PDF (adapted grid<DIM, T> x number of samples)

	Logger SimLog(time_vector.size() - 1);				// Simulation logger

	INT Random_Samples = 1;
	INT aux_Samples = 0;

	for (UINT i = 0; i < PARAM_DIMENSIONS; i++) {
		Random_Samples *= n_Samples[i];
		aux_Samples += n_Samples[i];
	}

	// GPU arrays
	thrust::device_vector<INT>		GPU_nSamples(n_Samples, n_Samples + PARAM_DIMENSIONS);	// Small vector containing the number of samples per parameter
	std::cout << "Total number of random samples: " << Random_Samples << ".\n";

	thrust::device_vector<T>					GPU_PDF = *H_PDF;														// PDF values at fixed Grid Nodes (for the GPU)

	thrust::device_vector<gridPoint<DIM, T>>	GPU_Part_Position;														// Particle positions (for the GPU)
	thrust::device_vector<T>					GPU_AdaptPDF;															// PDF value at Particle positions (for the GPU)

	thrust::device_vector<Param_pair>			GPU_Parameter_Mesh(Parameter_Mesh, Parameter_Mesh + aux_Samples);		// Parameter Problem_Domain array (for the GPU)

	// auxiliary variable that will be used for ensemble mean computation
	T Sum_Rand_Params = 0;
	for (UINT i = 0; i < Random_Samples; i++) {
		Param_vec aux_PM = Gather_Param_Vec(i, Parameter_Mesh, n_Samples);
		Sum_Rand_Params += aux_PM.Joint_PDF;
	}

	const UINT MAX_MEMORY_USABLE = 0.95 * prop->totalGlobalMem;		// max memory to be used (in bytes). 95% just in case

	// ------------------ DEFINITION OF THE INTERPOLATION VARIABLES AND ARRAYS ------------------ //
	UINT Adapt_Points, MaxNeighborNum;

	const UINT	max_steps 	  = 1000;		 								// max steps at the Conjugate Gradient (CG) algorithm
	const T 	in_tolerance  = TOLERANCE_ConjGrad, 
				search_radius = DISC_RADIUS * Problem_Domain.Discr_length();// CG stop tolerance and max radius to search ([3,6] appears to be optimal)

	// Now we make a slightly larger domain for the computations:
	grid<DIMENSIONS, TYPE> Base_Mesh;

	const UINT expansion_nodes = 40;									// 40 nodes expansion by each side
	Base_Mesh.Expand_From(Problem_Domain, expansion_nodes);				// From the initial Problem domain, we create an expanded version (with the same discretization!)

	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// ---------------------------------- SIMULATION TIME!! ---------------------------------------
	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------

	// -------------------- Store the 1st PDF (Init. Cond.) -----------------------------------
	thrust::copy(H_PDF->begin(), H_PDF->end(), &(*store_PDFs)[0]);

	// ------------------------------------------------------------------------------------
	uint16_t j = 0, mode = 0; 	// iteration counter and variable that modifies the vector field to go between the unforced and forced fields
	int16_t error_check = 0;	// auxiliary variable for error checking

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
	while (j < time_vector.size() - 1 && error_check != -1) {

		double	t0 = time_vector[j].time, tF = time_vector[j + 1].time;

		std::cout << "+---------------------------------------------------------------------+\n";
		// 1.- Initial step Adaptive Problem_Domain Refinement. First store the initial PDF with AMR performed

	auto startTimeSeconds = std::chrono::high_resolution_clock::now();

		error_check = ADAPT_MESH_REFINEMENT_nD<DIM, T>(*H_PDF, GPU_PDF, AdaptPDF, AdaptGrid, Problem_Domain, Base_Mesh, Supp_BBox);
		if (error_check == -1) { break; }

	auto endTimeSeconds = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration_3 = endTimeSeconds - startTimeSeconds;

		// Enter the information into the log information
		SimLog.subFrame_time[5*j] = duration_3.count();
		
		// Number of particles to advect
		Adapt_Points = AdaptGrid.size();

		GPU_PDF.clear();

		// Send adapted values and points to the GPU
		GPU_AdaptPDF.resize(Adapt_Points);
		GPU_Part_Position.resize(Adapt_Points);

		GPU_AdaptPDF 		= AdaptPDF;
		GPU_Part_Position 	= AdaptGrid;

		// Here, we're going to interpolate!!
				// ----------------------------------------------------------------------------------- //
				// -------------------------- INTERPOLATION ------------------------------------------ //
				// ----------------------------------------------------------------------------------- //
				
				// Determine threads and blocks for the simulation
				uint16_t	Threads = fmin(THREADS_P_BLK, Adapt_Points);
				UINT		Blocks	= floor((Adapt_Points - 1) / Threads) + 1;

				// Maximum neighbors to search. Diameter number of points powered to the dimension
				MaxNeighborNum = round(0.8 * fmin(pow(2 * round(DISC_RADIUS) + 1, DIM), Adapt_Points));

				// ------------------ RESIZING OF THE INTERPOLATION MATRIX ------------------ //
					thrust::device_vector<INT>	GPU_Index_array(MaxNeighborNum * Adapt_Points, -1);
					thrust::device_vector<T>	GPU_Mat_entries(MaxNeighborNum * Adapt_Points, 0);
					thrust::device_vector<UINT>	GPU_Num_Neighbors(Adapt_Points, 0);
				// -------------------------------------------------------------------------- //
				// 1.- Build Matix in GPU (indexes, dists and neighbors) Using Exahustive search...
				startTimeSeconds = std::chrono::high_resolution_clock::now();

				// Dynamical choice of either exhaustive or counting sort-based point search
				if (Adapt_Points < ptSEARCH_THRESHOLD) {
					Exh_PP_Search<DIM,T> << <Blocks, Threads >> > (	rpc(GPU_Part_Position, 0),
																	rpc(GPU_Part_Position, 0),
																	rpc(GPU_Index_array, 0),
																	rpc(GPU_Mat_entries, 0),
																	rpc(GPU_Num_Neighbors, 0),
																	MaxNeighborNum,
																	Adapt_Points,
																	Adapt_Points,
																	search_radius);
					gpuError_Check(cudaDeviceSynchronize());
				}
				else {
					error_check = _CS_Neighbor_Search<DIM, T>(	GPU_Part_Position,
																GPU_AdaptPDF,
																GPU_Index_array,
																GPU_Mat_entries,
																GPU_Num_Neighbors,
																Adapt_Points,
																MaxNeighborNum,
																Supp_BBox,
																search_radius);

					if (error_check == -1) { break; }
				}

				endTimeSeconds = std::chrono::high_resolution_clock::now();
				duration_3 = endTimeSeconds - startTimeSeconds;
				
				// Enter the information into the log information
				SimLog.subFrame_time[5*j + 2] = duration_3.count();


				// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
				thrust::device_vector<T>	GPU_lambdas(Adapt_Points);		// solution vector (RBF weights)
				thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0.0f);	// this will serve as the initial condition

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				error_check = CONJUGATE_GRADIENT_SOLVE<T>(GPU_lambdas,
															GPU_Index_array,
															GPU_Mat_entries,
															GPU_Num_Neighbors,
															GPU_AdaptPDF,
															Adapt_Points,
															MaxNeighborNum,
															max_steps,
															in_tolerance);
				if (error_check == -1) { std::cout << "Convergence failure.\n"; break; }
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				duration_3 = endTimeSeconds - startTimeSeconds;
				
				// Enter the information into the log information
				SimLog.subFrame_time[5*j + 3] = duration_3.count();
				SimLog.ConvergenceIterations[j] = error_check;

				// Clear them, to save memory
				GPU_Index_array.clear();
				GPU_Mat_entries.clear();
				GPU_Num_Neighbors.clear();

				// // ----------------------------------------------------------------------------------- //
				// // THIS PART ONLY GRABS THE LAST "OPTIMAL" LAMBDA AND COMPUTES ITS "PROJECTION" INTO THE SUBSPACE
				
				// if (DIM < 3) {
				// 	T temp = thrust::reduce(thrust::device, GPU_lambdas.begin(), GPU_lambdas.end());
				// 	thrust::transform(GPU_lambdas.begin(), GPU_lambdas.end(), GPU_lambdas.begin(), 1.0f / temp * _1);
				// }
				// // ----------------------------------------------------------------------------------- //


		// 1.1.- COMPUTE THE TRANSFORMATION OF THE PDF (IF THERE IS ONE)
		if (time_vector[j].impulse) {

		#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-T IMPULSE!

			std::cout << "RVT transformation at time: " << t0 << "\n";

			startTimeSeconds = std::chrono::high_resolution_clock::now();

			error_check = IMPULSE_TRANSFORM_PDF<DIM, T>(AdaptGrid,
														H_PDF,
														GPU_PDF,
														AdaptPDF,
														time_vector[j],
														jump,
														Problem_Domain,
														Base_Mesh,
														Supp_BBox);

			endTimeSeconds = std::chrono::high_resolution_clock::now();
			duration_3 = endTimeSeconds - startTimeSeconds;

			
			// Enter the information into the log information
			SimLog.subFrame_time[5*j + 1] = duration_3.count();

			assert(error_check == 0);

			AdaptGrid.clear();
			AdaptPDF.clear();

			jump++;
			j++;

			// Store info in cumulative variable
			thrust::copy(H_PDF->begin(), H_PDF->end(), &(*store_PDFs)[j * Problem_Domain.Total_Nodes()]);

		#elif(IMPULSE_TYPE == 2)	// THIS IS FOR HEAVISIDE-T IMPULSE!
			mode++;
			std::cout << "Now the vector field is in mode: " << mode % 2 << ".\n";

			AdaptGrid.clear();
			AdaptPDF.clear();

			j++;

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
			const UINT mem_requested_per_sample = (UINT)Adapt_Points * (sizeof(T) * 2 + sizeof(gridPoint<DIM, T>));

			// Set number of random samples to work with, and number of blocks to use
			UINT Random_Samples_Blk_size = (UINT)fmin((UINT)Random_Samples, MAX_MEMORY_USABLE / mem_requested_per_sample);
			UINT total_blocks = (UINT)ceilf((float)Random_Samples / Random_Samples_Blk_size);

			for (UINT b = 0; b < total_blocks; b++) {

				// Parameter sample offset to account for the block position
				UINT Sample_idx_offset_init = b * Random_Samples_Blk_size;
				UINT Sample_idx_offset_final = fmin((b + 1) * Random_Samples_Blk_size, Random_Samples);

				Random_Samples_Blk_size = Sample_idx_offset_final - Sample_idx_offset_init;

				UINT Block_Particles = Random_Samples_Blk_size * Adapt_Points;

				// 1.2.- Append the optimal particles once per sample!
				Full_AdaptGrid.resize(Block_Particles);
				Full_AdaptPDF .resize(Block_Particles);

				for (UINT k = 0; k < Random_Samples_Blk_size; k++) {
					std::copy(AdaptGrid.begin(), AdaptGrid.end(), &Full_AdaptGrid[k * Adapt_Points]);
					std::copy(AdaptPDF.begin(), AdaptPDF.end(), &Full_AdaptPDF[k * Adapt_Points]);
				}

				// Upload information to the GPU
				GPU_AdaptPDF.resize(Random_Samples_Blk_size);
				GPU_Part_Position.resize(Random_Samples_Blk_size);

				GPU_AdaptPDF 		= Full_AdaptPDF;
				GPU_Part_Position 	= Full_AdaptGrid;

				// Clear CPU info about the AMR procedure: no longer needed
				Full_AdaptGrid.clear();
				Full_AdaptPDF.clear();

				std::cout << "Size of relevant PDF points (per sample): " << Adapt_Points << "\n";	// this allows to check if the info is passed to the GPU correctly

				// Determine threads and blocks for the simulation
				Threads = fmin(THREADS_P_BLK, Block_Particles);
				Blocks	= floor((Block_Particles - 1) / Threads) + 1;

				// ------------------------------------------------------------------------------------ //
				// -------------------------- POINT ADVECTION ----------------------------------------- //
				// ------------------------------------------------------------------------------------ //
				startTimeSeconds = std::chrono::high_resolution_clock::now();
				ODE_INTEGRATE<DIM, T><< <Blocks, Threads >> > (	rpc(GPU_Part_Position, 0),
																rpc(GPU_AdaptPDF, 0),
																rpc(GPU_lambdas, 0),
																rpc(GPU_Parameter_Mesh, Sample_idx_offset_init),
																rpc(GPU_nSamples, 0),
																t0,
																deltaT,
																tF,
																Adapt_Points,
																Random_Samples_Blk_size,
																mode,
																rpc(Extra_Parameter, 0),
																Problem_Domain);
				gpuError_Check(cudaDeviceSynchronize()); // Here, the entire Problem_Domain points (those that were selected) and PDF points (same) have been updated.

				endTimeSeconds = std::chrono::high_resolution_clock::now();
				duration_3 = endTimeSeconds - startTimeSeconds;

				// Enter the information into the log information
				SimLog.subFrame_time[5*j + 1] = duration_3.count();

				// Before going to the next step, define the bounding box of the advected particles!
					thrust::device_vector<T> projection(Block_Particles,(T)0);

					for (uint16_t d = 0; d < DIM; d++) {
						findProjection<DIM, T> << <Blocks, Threads >> > (rpc(GPU_Part_Position, 0), rpc(projection, 0), Block_Particles, d);

						T temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end())); // min element from the projection in that direction
						T temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));


						// Eliminate the need for it to be in the Problem domain
						Supp_BBox.Boundary_inf.dim[d] = temp_1 - ceil(DISC_RADIUS) * Problem_Domain.Discr_length();
						Supp_BBox.Boundary_sup.dim[d] = temp_2 + ceil(DISC_RADIUS) * Problem_Domain.Discr_length();
					}

			
			// ----------------------------------------------------------------------------------- //
			// THIS PART ONLY GRABS THE LAST "OPTIMAL" LAMBDA AND COMPUTES ITS "PROJECTION" INTO THE SUBSPACE
			
			if (DIM < 3) {
				T temp = thrust::reduce(thrust::device, GPU_AdaptPDF.begin(), GPU_AdaptPDF.end());
				thrust::transform(GPU_AdaptPDF.begin(), GPU_AdaptPDF.end(), GPU_AdaptPDF.begin(), Random_Samples_Blk_size / temp * _1);
			}
			// ----------------------------------------------------------------------------------- //

			// 3.- Multiplication of matrix-lambdas to obtain updated grid<DIM, T> nodes
				// I'M GOING TO FIND THE NEAREST GRID NODES TO EACH PARTICLE

				GPU_PDF.resize(Problem_Domain.Total_Nodes(), 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
				Threads = fminf(THREADS_P_BLK, Block_Particles);
				Blocks  = floorf((Block_Particles - 1) / Threads) + 1;

				startTimeSeconds = std::chrono::high_resolution_clock::now();
				RESTART_GRID_FIND_GN<DIM,T> << < Blocks, Threads >> > (	rpc(GPU_Part_Position, 0),
																		rpc(GPU_PDF, 0),
																		rpc(GPU_AdaptPDF, 0),
																		rpc(GPU_Parameter_Mesh, 0),
																		rpc(GPU_nSamples, 0),
																		search_radius,
																		Adapt_Points,
																		Random_Samples_Blk_size,
																		Sample_idx_offset_init,
																		Problem_Domain,
																		Base_Mesh);
				gpuError_Check(cudaDeviceSynchronize());
				endTimeSeconds = std::chrono::high_resolution_clock::now();
				duration_3 = endTimeSeconds - startTimeSeconds;

				
				// Enter the information into the log information
				SimLog.subFrame_time[5 * j + 4] = duration_3.count();
			}

			AdaptGrid.clear();
			AdaptPDF.clear();
			// Correction of any possible negative PDF values
			// Re-define Threads and Blocks
			UINT Threads = fminf(THREADS_P_BLK, Problem_Domain.Total_Nodes());
			UINT Blocks = floorf((Problem_Domain.Total_Nodes() - 1) / Threads) + 1;

			CORRECTION<T> << <Blocks, Threads >> > (rpc(GPU_PDF, 0), Problem_Domain.Total_Nodes());
			gpuError_Check(cudaDeviceSynchronize());

			thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.0f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)

			*H_PDF = GPU_PDF; // Send back to CPU

			SimLog.writeToCLI(OUTPUT_INFO, j);

			std::cout << "+---------------------------------------------------------------------+\n";

			j++;

			// Store info in cumulative variable
			thrust::copy(H_PDF->begin(), H_PDF->end(), &(*store_PDFs)[j * Problem_Domain.Total_Nodes()]);
		}
	}

	SimLog.writeToFile();

	return error_check;
}
#endif