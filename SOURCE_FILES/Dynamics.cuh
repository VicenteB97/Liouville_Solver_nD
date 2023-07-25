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

#ifndef __DYNAMICS_CUH__
#define __DYNAMICS_CUH__

#include "Classes.cuh"
#include "Simulation_parameters.cuh"
#include "Parameter_distributions.cuh"
#include "Adapt_Mesh.cuh"
#include "Mat_Ops.cuh"
#include "Impulse_transformations.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

__device__ inline gridPoint VECTOR_FIELD(gridPoint X, TYPE t, const Param_vec parameter, const uint32_t mode, const FIXED_TYPE extra_param[]) {
	gridPoint output = VEC_FIELD;
	
	return output;
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__ inline TYPE DIVERGENCE_FIELD(gridPoint X, TYPE t, const Param_vec parameter, const uint32_t mode, const FIXED_TYPE extra_param[]) {
	TYPE output = DIVERGENCE;
	
	return output;
}
// -------------------------------------------------------------------------//
// -------------------------------------------------------------------------//
// -------------------------------------------------------------------------//
// -------------------------------------------------------------------------//

//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
// This part contains the definitions the functions that will be used to evolve the PDF from its initial state //
// We STRONGLY RECOMMEND not changing absolutely anything if your only use of this program is to simulate	   //
//-------------------------------------------------------------------------------------------------------------//


//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------//
//------------------------ MAIN COMPUTATION FUNCTIONS -----------------------------------------//
//---------------------------------------------------------------------------------------------//
// 1.- Runge-Kutta integrator (particle + value) ----------------------------------------------//
// 2.- Iterations of the Liouville-Equation solver --------------------------------------------//
//---------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------//

/// @brief This function computes the advection of the particles created by AMR.
/// 	It uses the RK4 scheme for the spatial variables advection and the Simpson rule for the exponential integral.
/// @param Particles Particle location (spatial variables)
/// @param PDF PDF value at the corresponding particle location
/// @param parameters Parameters to be used for the vector field and its corresponding divergence function
/// @param t0 Inital time for starting the simulation
/// @param deltaT Time step used in the simulation
/// @param ReinitSteps Number of steps before needing a re-interpolation
/// @param Adapt_Points Number of particles as computed by the AMR scheme
/// @param Random_Samples Number of random parameter samples
/// @return 
__global__ void RungeKutta(	gridPoint* 			Particles,
							TYPE* 				PDF,
							const Param_pair* 	parameters,
							const int32_t* 		n_Samples,
							float				t0,
							const float			deltaT,
							const float			tF,
							const int32_t		Adapt_Points,
							const int32_t		Random_Samples,
							const uint32_t		mode,
							const FIXED_TYPE*		extra_param,
							const gridPoint*	Domain_boundary) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Adapt_Points * Random_Samples) {

		// AUXILIARY DATA TO RUN THE ITERATIONS
		// So, the total amount of advections are going to be: (no. particles x no. of samples)
		const uint32_t  i_sample 	= floorf((float) i / Adapt_Points);
		const Param_vec parameter 	= _Gather_Param_Vec(i_sample, parameters, n_Samples);

		gridPoint k0, k1, k2, k3, k_final, aux;
		TYPE	  Int1, Int2, Int3;

		gridPoint x0 	= Particles[i]; 	// register storing the initial particle loc.
		TYPE Int_PDF	= PDF[i];		// register storing the initial particle value

		while (t0 < tF - deltaT / 2) {
			// Particle flow
			k0 = VECTOR_FIELD(x0, t0, parameter, mode, extra_param);

			aux = Mult_by_Scalar(deltaT / 2.0f, k0);
			k1  = VECTOR_FIELD(x0 + aux, t0 + deltaT / 2.0f, parameter, mode, extra_param);

			aux = Mult_by_Scalar(deltaT / 2.0f, k1);
			k2  = VECTOR_FIELD(x0 + aux, t0 + deltaT / 2.0f, parameter, mode, extra_param);

			aux = Mult_by_Scalar(deltaT, k2);
			k3  = VECTOR_FIELD(x0 + aux, t0 + deltaT, parameter, mode, extra_param);

			k1 = Mult_by_Scalar(2.00f, k1);
			k2 = Mult_by_Scalar(2.00f, k2);

			aux = x0 + Mult_by_Scalar((TYPE) deltaT / 6.00f, (k0 + k3 + k1 + k2)); // New particle dim

			// Integration of PDF: The following line corresponds to computing the approximation via a Hermite interpolation (we know initial and final points and their velocities)
			Int1 = DIVERGENCE_FIELD(x0, t0, parameter, mode, extra_param);

			k_final = VECTOR_FIELD(aux, t0 + deltaT, parameter, mode, extra_param);
			x0 	 = Mult_by_Scalar(0.50f, (x0 + aux));
			x0 	 = x0 + Mult_by_Scalar(0.125f, (k0 + k_final));
			Int2 = DIVERGENCE_FIELD(x0, (TYPE)(2.00f * t0 + deltaT) / 2.00f, parameter, mode, extra_param);

			Int3 = DIVERGENCE_FIELD(aux, (TYPE)t0 + deltaT, parameter, mode, extra_param);

			Int_PDF *= expf((TYPE)-deltaT / 6.00f * (Int1 + 4.00f * Int2 + Int3)); // New particle value

			// Reinit step
			x0 = aux;
			t0 += deltaT;

			if(!__is_in_domain(aux, Domain_boundary)){Int_PDF = 0; break;}	// Condition is equivalent to the homogeneous Neumann condition
		}

		Particles[i] 	= aux;
		PDF[i] 			= Int_PDF;
	}
}


/// @brief This function computes the Liouville Eq. iterations from t0 to tF. Each iteration consists of the following steps:
/// 1st) Compute the AMR of the initial PDF. 
/// 2nd) Create as many particle families as random samples there are.
/// 3rd) Advect all particles via RungeKutta function. 
/// 4th) Interpolate all families of particles onto the underlying high-res. fixed grid. 
/// 5th) Compute ensemble mean.
/// 6th) Store final PDF into the iteration vector for further post-processing and/or evolution visualization.
/// @param store_PDFs 
/// @param Parameter_Mesh 
/// @param H_Mesh 
/// @param H_PDF 
/// @param LvlFine 
/// @param LvlCoarse 
/// @param PtsPerDim 
/// @param Grid_Nodes 
/// @param time_vector 
/// @param deltaT 
/// @param ReinitSteps 
/// @return 
int32_t PDF_ITERATIONS( cudaDeviceProp* prop,
						std::vector<float>* store_PDFs,
						const Param_pair* Parameter_Mesh,
						const gridPoint* H_Mesh,
						thrust::host_vector<TYPE>* H_PDF,
						const int32_t* n_Samples,
						const int32_t& LvlFine,
						const int32_t& LvlCoarse,
						const uint32_t& PtsPerDim,
						const uint32_t& Grid_Nodes,
						const std::vector<Time_Impulse_vec> time_vector,
						const FIXED_TYPE& deltaT,
						const uint32_t& ReinitSteps) {

//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
//------------------- INITIALIZATION OF VECTORS/ARRAYS THAT WILL BE USED ---------------------//
//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
	std::vector<gridPoint>	AdaptGrid(0);			// Particle positions to be used for simulation (corresponding dim from AMR)
	std::vector<TYPE>		AdaptPDF(0);			// PDF value at the particle positions (corresponding values from AMR)
	std::vector<gridPoint>	Full_AdaptGrid(0);		// Final adapted grid (adapted grid x number of samples)
	std::vector<TYPE>		Full_AdaptPDF(0);		// Final adapted PDF (adapted grid x number of samples)

	int32_t Random_Samples  = 1;
	int32_t aux_Samples 	= 0;

	for (uint16_t i = 0; i < PARAM_DIMENSIONS; i++){
		Random_Samples 	*= n_Samples[i];
		aux_Samples 	+= n_Samples[i];
	}
	
	std::cout << "Total number of random samples: " << Random_Samples << ".\n";

	thrust::device_vector<gridPoint>	GPU_Part_Position;		// Particle positions (for the GPU)
	thrust::device_vector<TYPE>			GPU_AdaptPDF;			// PDF value at Particle positions (for the GPU)
	thrust::device_vector<Param_pair>	GPU_Parameter_Mesh(Parameter_Mesh, Parameter_Mesh + aux_Samples);		// Parameter H_Mesh array (for the GPU)

	thrust::device_vector<int32_t>		GPU_nSamples(n_Samples, n_Samples + PARAM_DIMENSIONS);
	thrust::device_vector<TYPE>			GPU_PDF = *H_PDF;					// PDF values at fixed Grid Nodes (for the GPU)

	// auxiliary variable that will be used for ensemble mean computation
	TYPE Sum_Rand_Params = 0;
	for (uint32_t i = 0; i < Random_Samples; i++) {
		Param_vec aux_PM = _Gather_Param_Vec(i, Parameter_Mesh, n_Samples);
		Sum_Rand_Params += aux_PM.Joint_PDF;
	}

	const uint64_t MAX_MEMORY_USABLE = 0.85*(prop->totalGlobalMem - aux_Samples * sizeof(Param_pair) - Grid_Nodes * sizeof(TYPE));		// max memory to be used in bytes


// ------------------ DEFINITION OF THE INTERPOLATION VARIABLES AND ARRAYS ------------------ //
	uint32_t Adapt_Points, MaxNeighborNum;

	const TYPE disc_X = (H_Mesh[1].dim[0] - H_Mesh[0].dim[0]);	// H_Mesh discretization size (per dimension)

	thrust::host_vector<gridPoint>	__H__Domain_Boundary(2);
	#pragma unroll
	for(uint16_t d = 0; d < DIMENSIONS; d++){
		__H__Domain_Boundary[0].dim[d] = H_Mesh[0].dim[d];
		__H__Domain_Boundary[1].dim[d] = H_Mesh[Grid_Nodes - 1].dim[d];
	}
	thrust::device_vector<gridPoint> __D__Domain_Boundary = __H__Domain_Boundary;

	const TYPE search_radius 	= DISC_RADIUS * disc_X;		// max radius to search ([3,6] appears to be optimal)

	const uint16_t	max_steps 		= 1000;		 				// max steps at the Conjugate Gradient (CG) algorithm
	const TYPE 	in_tolerance 	= TOLERANCE_ConjGrad; 		// CG stop tolerance

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// ---------------------------------- SIMULATION TIME!! ---------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

// -------------------- Store the 1st PDF (Init. Cond.) -----------------------------------
	thrust::copy(H_PDF->begin(),H_PDF->end(),&(*store_PDFs)[0]);

	// ------------------------------------------------------------------------------------
	uint32_t j = 0;				// PDF iteration counter
	uint32_t mode = 0; 		// this modifies the vector field to go between the unforced and square-forced field
	int32_t error_check = 0;	// auxiliary variable for error checking
	
	#if IMPULSE_TYPE == 1
		uint32_t jump = 0;	// auxiliary variable to know how many delta jumps have passed
	#endif
	
	#if INCLUDE_XTRA_PARAMS
		thrust::device_vector<FIXED_TYPE>	Extra_Parameter(XTRA_PARAM_LENGTH);
		thrust::copy(&XTRA_PARAM[0], &XTRA_PARAM[XTRA_PARAM_LENGTH], Extra_Parameter.begin());
	#else
		thrust::device_vector<FIXED_TYPE>	Extra_Parameter(0);
	#endif

	while (j < time_vector.size() - 1) {

		auto start_2 = std::chrono::high_resolution_clock::now();

		float	t0 = time_vector[j].time,
				tF = time_vector[j + 1].time;

		std::cout << "/ ------------------------------------------------------------------- /\n";
	// 1.- Initial step Adaptive H_Mesh Refinement. First store the initial PDF with AMR performed

	#if OUTPUT_INFO
		auto start_3 = std::chrono::high_resolution_clock::now();
		
		error_check = ADAPT_MESH_REFINEMENT_nD(*H_PDF, &GPU_PDF, &AdaptPDF, H_Mesh, &AdaptGrid, LvlFine, LvlCoarse, PtsPerDim);
		if(error_check == -1){return error_check;}

		auto end_3 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> duration_3 = end_3 - start_3;
		std::cout << "AMR iteration took " << duration_3.count() << " seconds\n";
	#else
		error_check = ADAPT_MESH_REFINEMENT_nD(*H_PDF, &GPU_PDF, &AdaptPDF, H_Mesh, &AdaptGrid, LvlFine, LvlCoarse, PtsPerDim);
		if(error_check == -1){return error_check;}
	#endif

	GPU_PDF.clear();

		// 1.1.- COMPUTE THE TRANSFORMATION OF THE PDF (IF THERE IS ONE)
		if (time_vector[j].impulse) {

			#if(IMPULSE_TYPE == 1)	// THIS IS FOR DELTA-TYPE IMPULSE!

				std::cout << "RVT transformation at time: " << t0 << "\n";

				#if OUTPUT_INFO
					start_3 = std::chrono::high_resolution_clock::now();
					
					error_check = IMPULSE_TRANSFORM_PDF(H_Mesh, 
													&AdaptGrid, 
													H_PDF, 
													&AdaptPDF, 
													time_vector[j],
													jump, 
													Grid_Nodes, 
													PtsPerDim,
													__D__Domain_Boundary);
					
					end_3 = std::chrono::high_resolution_clock::now();
					duration_3 = end_3 - start_3;
					std::cout << "Delta Impulse took " << duration_3.count() << " seconds\n";
				#else
					error_check = IMPULSE_TRANSFORM_PDF(H_Mesh, 
													&AdaptGrid, 
													H_PDF, 
													&AdaptPDF, 
													time_vector[j],
													jump, 
													Grid_Nodes, 
													PtsPerDim,
													__D__Domain_Boundary);
				#endif

				

				if (error_check != 0) {
					//std::cout << "Something went wrong...\n";
					std::cin.get();
					return error_check;
				}
				else {
					//std::cout << "Transformation done...continuing with Liouville PDE\n";
					AdaptGrid.clear();
					AdaptPDF.clear();
					
					jump++;
					j++;

					// Store info in cumulative variable
					thrust::copy(H_PDF->begin(),H_PDF->end(),&(*store_PDFs)[j * Grid_Nodes]);
				}
			#elif(IMPULSE_TYPE == 2)	// THIS IS FOR HEAVISIDE-TYPE IMPULSE!
				mode++;
				std::cout << "Now the vector field is in mode: " << mode % 2 << ".\n";

				AdaptGrid.clear();
				AdaptPDF .clear();

				j++;
			#elif(IMPULSE_TYPE != 0)
				std::cout << "Error in 'Dynamics.cuh'. You are choosing an unavailable option. Go back to 'Case_definition.cuh' and re-check options for IMPULSE_TYPE.\n";
				return -1;
			#endif
		}
	// 1.2.- COMPUTE THE SMOOTH EVOLUTION VIA LIOUVILLE GIBBS/CONTINUITY EQUATION
		else {
			std::cout << "Simulation time: " << t0 << " to " << tF << "\n";
			
			// Number of particles to advect
			Adapt_Points 	= AdaptGrid.size();
			
			// maximum neighbors to search. Diameter number of points powered to the dimension
			MaxNeighborNum 	= fmin(pow(2 * DISC_RADIUS, DIMENSIONS), Adapt_Points);

			// Total memory requirements for next Liouville step
			const uint64_t mem_requested_per_sample = (uint64_t) Adapt_Points * (sizeof(TYPE) * (6 + MaxNeighborNum) + sizeof(gridPoint) + sizeof(int32_t)*(MaxNeighborNum + 1) );

			// Set number of random samples to work with, and number of blocks to use
			uint32_t Random_Samples_Blk_size 	= (uint32_t) fmin	((uint64_t)Random_Samples, MAX_MEMORY_USABLE / mem_requested_per_sample);
			uint32_t total_blocks 				= (uint32_t) ceilf	((float)		Random_Samples / Random_Samples_Blk_size);

			for (uint32_t b = 0; b < total_blocks; b++){

				// Parameter sample offset to account for the block position
				uint32_t Sample_idx_offset_init  = b * Random_Samples_Blk_size;
				uint32_t Sample_idx_offset_final = fmin((b + 1) * Random_Samples_Blk_size, Random_Samples);

				Random_Samples_Blk_size = Sample_idx_offset_final - Sample_idx_offset_init;

				uint64_t Block_Particles = Random_Samples_Blk_size * Adapt_Points;

				Full_AdaptGrid.resize(Block_Particles);
				Full_AdaptPDF .resize(Block_Particles);
				
				// 1.2.- Append the optimal particles once per sample!
				for (uint32_t k = 0; k < Random_Samples_Blk_size; k++) {
					std::copy(AdaptGrid.begin(),AdaptGrid.end(),&Full_AdaptGrid[k * Adapt_Points]);
					std::copy(AdaptPDF .begin(),AdaptPDF .end(),&Full_AdaptPDF [k * Adapt_Points]);
				}

				// Upload information to the GPU
				GPU_AdaptPDF 		= Full_AdaptPDF;
				GPU_Part_Position 	= Full_AdaptGrid;

				// Clear CPU info about the AMR procedure: no longer needed
				Full_AdaptGrid.clear();
				Full_AdaptPDF .clear();

				std::cout << "Size of relevant PDF points (per sample): " << Adapt_Points << "\n";	// this allows to check if the info is passed to the GPU correctly

				// ------------------ RESIZING OF THE INTERPOLATION MATRIX ------------------ //

				thrust::device_vector<int32_t>		GPU_Index_array(MaxNeighborNum * Block_Particles);
				thrust::device_vector<TYPE>			GPU_Mat_entries(MaxNeighborNum * Block_Particles);
				thrust::device_vector<int32_t>		GPU_Num_Neighbors(Block_Particles);
				// -------------------------------------------------------------------------- //
				// Determine threads and blocks for the simulation
				uint16_t Threads = fmin(THREADS_P_BLK, Block_Particles);
				uint64_t Blocks  = floor((Block_Particles - 1) / Threads) + 1;

				// ------------------------------------------------------------------------------------ //
				// -------------------------- POINT ADVECTION ----------------------------------------- //
				// ------------------------------------------------------------------------------------ //
				// Using RK4 for time integration of characteristic curves
				#if OUTPUT_INFO
				start_3 = std::chrono::high_resolution_clock::now();
						
				RungeKutta << <Blocks, Threads >> >(rpc(GPU_Part_Position,0), 
													rpc(GPU_AdaptPDF,0),
													rpc(GPU_Parameter_Mesh, Sample_idx_offset_init), 
													rpc(GPU_nSamples,0), 
													t0, 
													deltaT, 
													tF, 
													Adapt_Points, 
													Random_Samples_Blk_size,
													mode,
													rpc(Extra_Parameter,0),
													rpc(__D__Domain_Boundary,0));
				gpuError_Check(cudaDeviceSynchronize()); // Here, the entire H_Mesh points (those that were selected) and PDF points (same) have been updated.
						
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;
				std::cout << "Runge-Kutta iteration took " << duration_3.count() << " seconds\n";
				#else
				RungeKutta << <Blocks, Threads >> >(rpc(GPU_Part_Position,0), 
													rpc(GPU_AdaptPDF,0),
													rpc(GPU_Parameter_Mesh,Sample_idx_offset_init), 
													rpc(GPU_nSamples,0), 
													t0, 
													deltaT, 
													tF, 
													Adapt_Points, 
													Random_Samples_Blk_size,
													mode,
													rpc(Extra_Parameter,0),
													rpc(__D__Domain_Boundary,0));
				gpuError_Check(cudaDeviceSynchronize()); // Here, the entire H_Mesh points (those that were selected) and PDF points (same) have been updated.
				#endif

				// ----------------------------------------------------------------------------------- //
				// -------------------------- INTERPOLATION ------------------------------------------ //
				// ----------------------------------------------------------------------------------- //
				// 1.- Build Matix in GPU (indexes, dists and neighbors) Using Exahustive search...
				#if OUTPUT_INFO
				start_3 = std::chrono::high_resolution_clock::now();
				
				Exh_PP_Search<TYPE> << <Blocks, Threads >> > (rpc(GPU_Part_Position,0), 
																rpc(GPU_Part_Position,0), 
																rpc(GPU_Index_array,0), 
																rpc(GPU_Mat_entries,0), 
																rpc(GPU_Num_Neighbors,0), 
																MaxNeighborNum, 
																Adapt_Points, 
																Block_Particles, 
																search_radius,
																rpc(__D__Domain_Boundary,0)); 
				gpuError_Check(cudaDeviceSynchronize());
						
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;
				std::cout << "Exhaustive point serach took " << duration_3.count() << " seconds\n";
				#else
				Exh_PP_Search<TYPE> << <Blocks, Threads >> > (rpc(GPU_Part_Position,0), 
														rpc(GPU_Part_Position,0), 
														rpc(GPU_Index_array,0), 
														rpc(GPU_Mat_entries,0), 
														rpc(GPU_Num_Neighbors,0), 
														MaxNeighborNum, 
														Adapt_Points, 
														Block_Particles, 
														search_radius,
														rpc(__D__Domain_Boundary,0)); 
				gpuError_Check(cudaDeviceSynchronize());
				#endif

				// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
				thrust::device_vector<TYPE>	GPU_lambdas(Block_Particles);	// solution vector (RBF weights)
				thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0.0f);		// this will serve as the initial condition

				#if OUTPUT_INFO
				start_3 = std::chrono::high_resolution_clock::now();

				error_check = CONJUGATE_GRADIENT_SOLVE<TYPE>(  GPU_lambdas, 
																GPU_Index_array, 
																GPU_Mat_entries, 
																GPU_Num_Neighbors, 
																GPU_AdaptPDF, 
																Block_Particles, 
																MaxNeighborNum, 
																max_steps, 
																in_tolerance);
				if (error_check == -1) { return error_check; }
						
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;
				std::cout << "Conjugate Gradient took " << duration_3.count() << " seconds\n";
				#else

				error_check = CONJUGATE_GRADIENT_SOLVE<TYPE>(  GPU_lambdas, 
																GPU_Index_array, 
																GPU_Mat_entries, 
																GPU_Num_Neighbors, 
																GPU_AdaptPDF, 
																Block_Particles, 
																MaxNeighborNum, 
																max_steps, 
																in_tolerance);
				if (error_check == -1) { return error_check; }
				#endif

				// ----------------------------------------------------------------------------------- //
				// THIS PART ONLY GRABS THE LAST "OPTIMAL" LAMBDA AND COMPUTES ITS "PROJECTION" INTO THE SUBSPACE
				#if DIMENSIONS == 2
					TYPE temp = thrust::reduce(thrust::device,GPU_lambdas.begin(),GPU_lambdas.end());
					thrust::transform(GPU_lambdas.begin(),GPU_lambdas.end(),GPU_lambdas.begin(), Random_Samples_Blk_size / temp * _1);
				#endif
				// ----------------------------------------------------------------------------------- //

			// 3.- Multiplication of matrix-lambdas to obtain updated grid nodes
				// I'M GOING TO FIND THE NEAREST GRID NODES TO EACH PARTICLE
				
				GPU_PDF.resize(Grid_Nodes, 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step

				#if OUTPUT_INFO
				start_3 = std::chrono::high_resolution_clock::now();

				Threads = fminf(THREADS_P_BLK, Block_Particles);
				Blocks  = floorf((Block_Particles - 1) / Threads) + 1;
				RESTART_GRID_FIND_GN<TYPE><<< Blocks, Threads >>>(	rpc(GPU_Part_Position,0),
																	rpc(GPU_PDF,0),
																	rpc(GPU_lambdas,0),
																	rpc(GPU_Parameter_Mesh,0),
																	rpc(GPU_nSamples,0),
																	search_radius,
																	H_Mesh[0],
																	disc_X,
																	PtsPerDim,
																	Adapt_Points,
																	Random_Samples_Blk_size,
																	Sample_idx_offset_init,
																	rpc(__D__Domain_Boundary,0));
				gpuError_Check(cudaDeviceSynchronize());
						
				end_3 = std::chrono::high_resolution_clock::now();
				duration_3 = end_3 - start_3;
				std::cout << "Remeshing took " << duration_3.count() << " seconds\n";

				#else
				Threads = fminf(THREADS_P_BLK, Block_Particles);
				Blocks  = floorf((Block_Particles - 1) / Threads) + 1;
				RESTART_GRID_FIND_GN<TYPE><<< Blocks, Threads >>>(	rpc(GPU_Part_Position,0),
																	rpc(GPU_PDF,0),
																	rpc(GPU_lambdas,0),
																	rpc(GPU_Parameter_Mesh,0),
																	rpc(GPU_nSamples,0),
																	search_radius,
																	H_Mesh[0],
																	disc_X,
																	PtsPerDim,
																	Adapt_Points,
																	Random_Samples_Blk_size,
																	Sample_idx_offset_init,
																	rpc(__D__Domain_Boundary,0));
				gpuError_Check(cudaDeviceSynchronize());
				#endif
			}

			
			AdaptGrid.clear();
			AdaptPDF .clear();
			// Correction of any possible negative PDF values
			// Re-define Threads and Blocks
			uint16_t Threads = fminf(THREADS_P_BLK, Grid_Nodes);
			uint64_t Blocks  = floorf((Grid_Nodes - 1) / Threads) + 1;
			CORRECTION<TYPE><<<Blocks, Threads>>>(rpc(GPU_PDF,0), Grid_Nodes);
			gpuError_Check(cudaDeviceSynchronize());

			thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.0f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)

			*H_PDF = GPU_PDF; // Send back to CPU
			j++;

			auto end_2 = std::chrono::high_resolution_clock::now();

			std::chrono::duration<float> duration_2 = end_2 - start_2;
			std::cout << "Liouville iteration took " << duration_2.count() << " seconds\n";
			std::cout << "/ ------------------------------------------------------------------- /\n";

			// Store info in cumulative variable
			thrust::copy(H_PDF->begin(),H_PDF->end(),&(*store_PDFs)[j * Grid_Nodes]);
		}
	}
	return error_check;
}
#endif