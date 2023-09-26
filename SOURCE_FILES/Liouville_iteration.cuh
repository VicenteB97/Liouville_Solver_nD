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
/// 4th) Interpolate all families of particles onto the underlying high-res. fixed grid. 
/// 5th) Compute ensemble mean.
/// 6th) Store final PDF into the iteration vector for further post-processing and/or evolution visualization.
/// @param store_PDFs 
/// @param Parameter_Mesh 
/// @param Base_Mesh 
/// @param H_PDF 
/// @param LvlFine 
/// @param LvlCoarse 
/// @param PtsPerDim 
/// @param Base_Mesh.Total_Nodes() 
/// @param time_vector 
/// @param deltaT 
/// @param ReinitSteps 
/// @return 
int16_t PDF_ITERATIONS(	cudaDeviceProp*				prop,
						std::vector<float>*			store_PDFs,
						const Param_pair*			Parameter_Mesh,
						const grid&					Base_Mesh,
						thrust::host_vector<TYPE>*	H_PDF,
						const INT*				n_Samples,
						const INT&				LvlFine,
						const INT&				LvlCoarse,
						const std::vector<Time_Impulse_vec> time_vector,
						const FIXED_TYPE&			deltaT,
						const UINT&				ReinitSteps) {

	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	//------------------- INITIALIZATION OF VECTORS/ARRAYS THAT WILL BE USED ---------------------//
	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	//--------------------------------------------------------------------------------------------//
	std::vector<gridPoint>	AdaptGrid;			// Particle positions to be used for simulation (corresponding dim from AMR)
	std::vector<TYPE>		AdaptPDF;			// PDF value at the particle positions (corresponding values from AMR)
	std::vector<gridPoint>	Full_AdaptGrid;		// Final adapted grid (adapted grid x number of samples)
	std::vector<TYPE>		Full_AdaptPDF;		// Final adapted PDF (adapted grid x number of samples)

	grid Supp_BBox = Base_Mesh;		// Initialize the Support bounding box as the initial mesh.First iteration will be slower but...whatcha gonna do?

	INT Random_Samples = 1;
	INT aux_Samples = 0;

	for (UINT i = 0; i < PARAM_DIMENSIONS; i++) {
		Random_Samples *= n_Samples[i];
		aux_Samples += n_Samples[i];
	}

	std::cout << "Total number of random samples: " << Random_Samples << ".\n";

	thrust::device_vector<gridPoint>	GPU_Part_Position;														// Particle positions (for the GPU)
	thrust::device_vector<TYPE>			GPU_AdaptPDF;															// PDF value at Particle positions (for the GPU)
	thrust::device_vector<Param_pair>	GPU_Parameter_Mesh(Parameter_Mesh, Parameter_Mesh + aux_Samples);		// Parameter Base_Mesh array (for the GPU)

	thrust::device_vector<INT>			GPU_nSamples(n_Samples, n_Samples + PARAM_DIMENSIONS);
	thrust::device_vector<TYPE>			GPU_PDF = *H_PDF;														// PDF values at fixed Grid Nodes (for the GPU)

	// auxiliary variable that will be used for ensemble mean computation
	TYPE Sum_Rand_Params = 0;
	for (UINT i = 0; i < Random_Samples; i++) {
		Param_vec aux_PM = _Gather_Param_Vec(i, Parameter_Mesh, n_Samples);
		Sum_Rand_Params += aux_PM.Joint_PDF;
	}

	const UINT MAX_MEMORY_USABLE = 0.9 * (prop->totalGlobalMem - aux_Samples * sizeof(Param_pair) - Base_Mesh.Total_Nodes() * sizeof(TYPE));		// max memory to be used in bytes


	// ------------------ DEFINITION OF THE INTERPOLATION VARIABLES AND ARRAYS ------------------ //
	UINT Adapt_Points, MaxNeighborNum;

	const UINT	max_steps = 1000;		 				// max steps at the Conjugate Gradient (CG) algorithm
	const TYPE 	in_tolerance  = TOLERANCE_ConjGrad, search_radius = DISC_RADIUS * Base_Mesh.Discr_length(); 		// CG stop tolerance and max radius to search ([3,6] appears to be optimal)

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
		thrust::device_vector<FIXED_TYPE>	Extra_Parameter(XTRA_PARAM_LENGTH);
		thrust::copy(&XTRA_PARAM[0], &XTRA_PARAM[XTRA_PARAM_LENGTH], Extra_Parameter.begin());
	#else
		thrust::device_vector<FIXED_TYPE>	Extra_Parameter(0);
	#endif


// IN THIS LINE WE COMMENCE WITH THE ACTUAL ITERATIONS OF THE LIOUVILLE EQUATION
	while (j < time_vector.size() - 1 && error_check == 0) {

		// ERASEEEEEEEEEEEEEEEEEEEEEE
		Supp_BBox = Base_Mesh;
		/////////////////////////////

	auto start_2 = std::chrono::high_resolution_clock::now();

		float	t0 = time_vector[j].time, tF = time_vector[j + 1].time;

		std::cout << "+---------------------------------------------------------------------+\n";
		// 1.- Initial step Adaptive Base_Mesh Refinement. First store the initial PDF with AMR performed

	auto start_3 = std::chrono::high_resolution_clock::now();

		error_check = ADAPT_MESH_REFINEMENT_nD(*H_PDF, GPU_PDF, AdaptPDF, AdaptGrid, Base_Mesh, Supp_BBox, LvlFine, LvlCoarse);
		if (error_check == -1) { break; }

	auto end_3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration_3 = end_3 - start_3;

		// Support bounding box reinitialization
		Supp_BBox.Boundary_inf = Base_Mesh.Boundary_sup;	// Reinitialize the inf. boundary node
		Supp_BBox.Boundary_sup = Base_Mesh.Boundary_inf;	// Reinitialize the sup. boundary node
		Supp_BBox.Nodes_per_Dim = Base_Mesh.Nodes_per_Dim;	// Reinitialize the nodes per dimension

		#if OUTPUT_INFO
		std::cout << "AMR iteration took " << duration_3.count() << " seconds\n";
		#endif

		// 1.1.- COMPUTE THE TRANSFORMATION OF THE PDF (IF THERE IS ONE)
		if (time_vector[j].impulse) {

	#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-TYPE IMPULSE!

			std::cout << "RVT transformation at time: " << t0 << "\n";

			start_3 = std::chrono::high_resolution_clock::now();

			error_check = IMPULSE_TRANSFORM_PDF(AdaptGrid,
												H_PDF,
												GPU_PDF,
												AdaptPDF,
												time_vector[j],
												jump,
												Base_Mesh,
												Supp_BBox);

			end_3 = std::chrono::high_resolution_clock::now();
			duration_3 = end_3 - start_3;

#if OUTPUT_INFO
			std::cout << "Delta Impulse took " << duration_3.count() << " seconds\n";
#endif

			assert(error_check == 0);

			AdaptGrid.clear();
			AdaptPDF.clear();

			jump++;
			j++;

			// Store info in cumulative variable
			thrust::copy(H_PDF->begin(), H_PDF->end(), &(*store_PDFs)[j * Base_Mesh.Total_Nodes()]);

#elif(IMPULSE_TYPE == 2)	// THIS IS FOR HEAVISIDE-TYPE IMPULSE!
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

			GPU_PDF.clear();

			// Number of particles to advect
			Adapt_Points = AdaptGrid.size();

			// maximum neighbors to search. Diameter number of points powered to the dimension
			MaxNeighborNum = fmin(pow(2 * round(DISC_RADIUS) + 1, DIMENSIONS), Adapt_Points);

			// Total memory requirements for next Liouville step
			const UINT mem_requested_per_sample = (UINT)Adapt_Points * (sizeof(TYPE) * (6 + MaxNeighborNum) + sizeof(gridPoint) + sizeof(INT) * (MaxNeighborNum + 1));

			// Set number of random samples to work with, and number of blocks to use
			UINT Random_Samples_Blk_size = (UINT)fmin((UINT)Random_Samples, MAX_MEMORY_USABLE / mem_requested_per_sample);
			UINT total_blocks = (UINT)ceilf((float)Random_Samples / Random_Samples_Blk_size);

			for (UINT b = 0; b < total_blocks; b++) {

				// Parameter sample offset to account for the block position
				UINT Sample_idx_offset_init = b * Random_Samples_Blk_size;
				UINT Sample_idx_offset_final = fmin((b + 1) * Random_Samples_Blk_size, Random_Samples);

				Random_Samples_Blk_size = Sample_idx_offset_final - Sample_idx_offset_init;

				UINT Block_Particles = Random_Samples_Blk_size * Adapt_Points;

				Full_AdaptGrid.resize(Block_Particles);
				Full_AdaptPDF.resize(Block_Particles);

				// 1.2.- Append the optimal particles once per sample!
				for (UINT k = 0; k < Random_Samples_Blk_size; k++) {
					std::copy(AdaptGrid.begin(), AdaptGrid.end(), &Full_AdaptGrid[k * Adapt_Points]);
					std::copy(AdaptPDF.begin(), AdaptPDF.end(), &Full_AdaptPDF[k * Adapt_Points]);
				}

				// Upload information to the GPU
				GPU_AdaptPDF = Full_AdaptPDF;
				GPU_Part_Position = Full_AdaptGrid;

				// Clear CPU info about the AMR procedure: no longer needed
				Full_AdaptGrid.clear();
				Full_AdaptPDF.clear();

				std::cout << "Size of relevant PDF points (per sample): " << Adapt_Points << "\n";	// this allows to check if the info is passed to the GPU correctly

				// Determine threads and blocks for the simulation
				uint16_t	Threads = fmin(THREADS_P_BLK, Block_Particles);
				UINT		Blocks	= floor((Block_Particles - 1) / Threads) + 1;

				// ------------------------------------------------------------------------------------ //
				// -------------------------- POINT ADVECTION ----------------------------------------- //
				// ------------------------------------------------------------------------------------ //
				start_3 = std::chrono::high_resolution_clock::now();
				ODE_INTEGRATE<< <Blocks, Threads >> > (	rpc(GPU_Part_Position, 0),
														rpc(GPU_AdaptPDF, 0),
														rpc(GPU_Parameter_Mesh, Sample_idx_offset_init),
														rpc(GPU_nSamples, 0),
														t0,
														deltaT,
														tF,
														Adapt_Points,
														Random_Samples_Blk_size,
														mode,
														rpc(Extra_Parameter, 0),
														Base_Mesh);
				gpuError_Check(cudaDeviceSynchronize()); // Here, the entire Base_Mesh points (those that were selected) and PDF points (same) have been updated.

				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;

				// Using RK4 for time integration of characteristic curves
				#if OUTPUT_INFO
					duration_3 = end_3 - start_3;
					std::cout << "Runge-Kutta iteration took " << duration_3.count() << " seconds\n";
				#endif

				// ----------------------------------------------------------------------------------- //
				// -------------------------- INTERPOLATION ------------------------------------------ //
				// ----------------------------------------------------------------------------------- //

				// ------------------ RESIZING OF THE INTERPOLATION MATRIX ------------------ //
					thrust::device_vector<INT>	GPU_Index_array(MaxNeighborNum* Block_Particles, -1);
					thrust::device_vector<TYPE>	GPU_Mat_entries(MaxNeighborNum* Block_Particles, 0);
					thrust::device_vector<UINT>	GPU_Num_Neighbors(Block_Particles, 0);
				// -------------------------------------------------------------------------- //
				// 1.- Build Matix in GPU (indexes, dists and neighbors) Using Exahustive search...
				start_3 = std::chrono::high_resolution_clock::now();

				// Dynamical choice of either exhaustive or counting sort-based point search
				if (Adapt_Points < ptSEARCH_THRESHOLD) {
					Exh_PP_Search<TYPE> << <Blocks, Threads >> > (	rpc(GPU_Part_Position, 0),
																	rpc(GPU_Part_Position, 0),
																	rpc(GPU_Index_array, 0),
																	rpc(GPU_Mat_entries, 0),
																	rpc(GPU_Num_Neighbors, 0),
																	MaxNeighborNum,
																	Adapt_Points,
																	Block_Particles,
																	search_radius,
																	Base_Mesh);
					gpuError_Check(cudaDeviceSynchronize());

					thrust::device_vector<TYPE> projection(Block_Particles);

					for (uint16_t d = 0; d < DIMENSIONS; d++) {
						findProjection<TYPE> << <Blocks, Threads >> > (rpc(GPU_Part_Position, 0), rpc(projection, 0), Block_Particles, d);

						TYPE temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end()));
						TYPE temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));

						Supp_BBox.Boundary_inf.dim[d] = fmax(Base_Mesh.Boundary_inf.dim[d], fmin(Supp_BBox.Boundary_inf.dim[d], temp_1));
						Supp_BBox.Boundary_inf.dim[d] = fmin(Base_Mesh.Boundary_sup.dim[d], fmax(Supp_BBox.Boundary_sup.dim[d], temp_2));
					}

				}
				else {
					error_check = _CS_Neighbor_Search<TYPE>(GPU_Part_Position,
															GPU_AdaptPDF,
															GPU_Index_array,
															GPU_Mat_entries,
															GPU_Num_Neighbors,
															Adapt_Points,
															MaxNeighborNum,
															search_radius,
															Base_Mesh,
															Supp_BBox);

					if (error_check == -1) { break; }
				}
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;

				#if OUTPUT_INFO
					std::cout << "Point serach took " << duration_3.count() << " seconds\n";
				#endif


				// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
				thrust::device_vector<TYPE>	GPU_lambdas(Block_Particles);	// solution vector (RBF weights)
				thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0.0f);	// this will serve as the initial condition

				start_3 = std::chrono::high_resolution_clock::now();
				error_check = CONJUGATE_GRADIENT_SOLVE<TYPE>(GPU_lambdas,
															GPU_Index_array,
															GPU_Mat_entries,
															GPU_Num_Neighbors,
															GPU_AdaptPDF,
															Block_Particles,
															MaxNeighborNum,
															max_steps,
															in_tolerance);
				if (error_check == -1) { break; }
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;
				
				#if OUTPUT_INFO
					std::cout << "Conjugate Gradient took " << duration_3.count() << " seconds\n";
				#endif

				// Clear them, to save memory
				GPU_Index_array.clear();
				GPU_Mat_entries.clear();
				GPU_Num_Neighbors.clear();

				// ----------------------------------------------------------------------------------- //
				// THIS PART ONLY GRABS THE LAST "OPTIMAL" LAMBDA AND COMPUTES ITS "PROJECTION" INTO THE SUBSPACE
				
				#if DIMENSIONS == 2
					TYPE temp = thrust::reduce(thrust::device, GPU_lambdas.begin(), GPU_lambdas.end());
					thrust::transform(GPU_lambdas.begin(), GPU_lambdas.end(), GPU_lambdas.begin(), Random_Samples_Blk_size / temp * _1);
				#endif
				// ----------------------------------------------------------------------------------- //

			// 3.- Multiplication of matrix-lambdas to obtain updated grid nodes
				// I'M GOING TO FIND THE NEAREST GRID NODES TO EACH PARTICLE

				GPU_PDF.resize(Base_Mesh.Total_Nodes(), 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
				Threads = fminf(THREADS_P_BLK, Block_Particles);
				Blocks  = floorf((Block_Particles - 1) / Threads) + 1;

				start_3 = std::chrono::high_resolution_clock::now();
				RESTART_GRID_FIND_GN<TYPE> << < Blocks, Threads >> > (	rpc(GPU_Part_Position, 0),
																		rpc(GPU_PDF, 0),
																		rpc(GPU_lambdas, 0),
																		rpc(GPU_Parameter_Mesh, 0),
																		rpc(GPU_nSamples, 0),
																		search_radius,
																		Adapt_Points,
																		Random_Samples_Blk_size,
																		Sample_idx_offset_init,
																		Base_Mesh);
				gpuError_Check(cudaDeviceSynchronize());
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;
#if OUTPUT_INFO
				std::cout << "Remeshing took " << duration_3.count() << " seconds\n";
#endif
			}

			AdaptGrid.clear();
			AdaptPDF.clear();
			// Correction of any possible negative PDF values
			// Re-define Threads and Blocks
			UINT Threads = fminf(THREADS_P_BLK, Base_Mesh.Total_Nodes());
			UINT Blocks = floorf((Base_Mesh.Total_Nodes() - 1) / Threads) + 1;

			CORRECTION<TYPE> << <Blocks, Threads >> > (rpc(GPU_PDF, 0), Base_Mesh.Total_Nodes());
			gpuError_Check(cudaDeviceSynchronize());

			thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.0f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)

			*H_PDF = GPU_PDF; // Send back to CPU
			j++;

			auto end_2 = std::chrono::high_resolution_clock::now();

			std::chrono::duration<float> duration_2 = end_2 - start_2;
			std::cout << "Total Liouville iteration took " << duration_2.count() << " seconds\n";
			std::cout << "+---------------------------------------------------------------------+\n";

			// Store info in cumulative variable
			thrust::copy(H_PDF->begin(), H_PDF->end(), &(*store_PDFs)[j * Base_Mesh.Total_Nodes()]);
		}
	}
	return error_check;
}
#endif