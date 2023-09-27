
//#if (IMPULSE_TYPE == 1) 	// this handles the examples for delta-like impulses
#ifndef __IMPULSE_TRANSFORMATIONS_CUH__
#define __IMPULSE_TRANSFORMATIONS_CUH__

#if IMPULSE_TYPE == 1
#include "Classes.cuh"
#include "Interpolation.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

int16_t RANDOMIZE_II(const INT* 					n_samples, 
				const INT 						Total_Samples, 
				std::vector<Impulse_Param_vec>* Parameter_Mesh, 
				const Distributions* 			Dist_Parameters);



/// <summary>
/// 
/// </summary>
/// <param name="Particle_Positions"></param>
/// <param name="impulse_strengths"></param>
/// <param name="Num_Particles_per_sample"></param>
/// <param name="Total_Particles"></param>
/// <returns></returns>
__global__ void TRANSFORM_PARTICLES(gridPoint<DIMENSIONS, TYPE>*					Particle_Positions,
									const Impulse_Param_vec*	impulse_strengths, 
									const INT					Num_Particles_per_sample,
									const INT					Total_Particles) {

	const INT i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles){

		const INT j =(INT) floorf((float)i / Num_Particles_per_sample);				// current sample
		Impulse_Param_vec aux = impulse_strengths[j];

		 
		for (uint16_t d = 0; d < DIMENSIONS; d++){			
			Particle_Positions[i].dim[d] += aux.sample_vec[d];		// we only have to add to the second variable!
		}

	}
}

/// <summary>
/// This function computes the new PDF after a certain impulse transform according to the Random Variable Transformation Theorem (RVT). To be linked with the Liouville PDE solver.
/// </summary>
/// <param name="MESH"></param>
/// <param name="Adapt_MESH"></param>
/// <param name="PDF"></param>
/// <param name="Adapt_PDF"></param>
int16_t IMPULSE_TRANSFORM_PDF(const std::vector<gridPoint<DIMENSIONS, TYPE>>&	Adapt_MESH,		// AMR-selected points
							thrust::host_vector<float>*		PDF,			// PDF in Mesh
							thrust::device_vector<float>&	GPU_PDF,			// PDF in Mesh
							const std::vector<float>&		Adapt_PDF,		// PDF in AMR-selected points
							const Time_Impulse_vec			time,			// time-impulse information 
							const INT						jump,			// current jump 
							const grid<DIMENSIONS, TYPE>&	Base_Mesh,
							grid<DIMENSIONS, TYPE>&			Supp_BBox){	 

// 0.- Create the impulse samples

		UINT Adapt_Points = Adapt_PDF.size();

		Distributions* Imp_Param_Dist = new Distributions[DIMENSIONS];
		INT n_samples[DIMENSIONS];

		 
		for (uint16_t d = 0; d < DIMENSIONS; d++){
				// RV mean and variance
				Imp_Param_Dist[d].Name  			= D_JUMP_DIST_NAMES[jump * DIMENSIONS + d];						// N, G or U distributions
				Imp_Param_Dist[d].Truncated  		= D_JUMP_DIST_TRUNC[jump * DIMENSIONS + d];						// TRUNCATED?
				Imp_Param_Dist[d].trunc_interval[0] = D_JUMP_DIST_InfTVAL[jump * DIMENSIONS + d];					// min of trunc. interval (if chosen low enough, automatically bounds to -6 std. deviations)
				Imp_Param_Dist[d].trunc_interval[1] = D_JUMP_DIST_SupTVAL[jump * DIMENSIONS + d]; 					// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
				Imp_Param_Dist[d].params[0] 		= D_JUMP_DIST_MEAN[jump * DIMENSIONS + d]; 	// mean
				Imp_Param_Dist[d].params[1] 		= D_JUMP_DIST_STD[jump * DIMENSIONS + d];		// std
				n_samples[d]						= D_JUMP_DIST_SAMPLES[jump * DIMENSIONS + d];			// no. of samples
		}

		INT Random_Samples = 1;
		 
		for (UINT i = 0; i < DIMENSIONS; i++){
			Random_Samples *= n_samples[i];
		}

		std::vector<Impulse_Param_vec> Impulse_Parameter_Mesh(Random_Samples);

		INT error_check = RANDOMIZE_II(n_samples, Random_Samples, &Impulse_Parameter_Mesh, Imp_Param_Dist);
		if (error_check == -1){return -1;}

		float Sum_Rand_Params = 0;
		for (INT i = 0; i < Random_Samples; i++) {
			Sum_Rand_Params += Impulse_Parameter_Mesh[i].Joint_PDF;
		}

		std::vector<gridPoint<DIMENSIONS, TYPE>>	Full_Adapt_Grid(0);
		std::vector<float>		Full_Adapt_PDF(0);

		for (INT i = 0; i < Random_Samples; i++) {
			Full_Adapt_Grid.insert(Full_Adapt_Grid.end(), Adapt_MESH.begin(), Adapt_MESH.end());
			Full_Adapt_PDF.insert(Full_Adapt_PDF.end(), Adapt_PDF.begin(), Adapt_PDF.end());
		}

		const INT Total_Particles = Full_Adapt_PDF.size();

// 1.- Do the transformation of the points according to the info given by the time-impulse vector
		thrust::device_vector<gridPoint<DIMENSIONS, TYPE>>			Particle_Positions	= Full_Adapt_Grid;		// To compute the transformations at the GPU
		thrust::device_vector<float>				PDF_Particles		= Full_Adapt_PDF;		// PDF of the transformations at the GPU
		thrust::device_vector<Impulse_Param_vec>	Impulses			= Impulse_Parameter_Mesh;

		INT Threads = (INT)fminf(THREADS_P_BLK, Total_Particles);
		INT Blocks  = (INT)floorf(Total_Particles / Threads) + 1;

	// FOR SOME REASON, I'M ONLY WRITING IN SOME VALUES...NOT ALL OF THEM

	TRANSFORM_PARTICLES << <Blocks, Threads >> > (	rpc(Particle_Positions,0), 
													rpc(Impulses,0), 
													Adapt_Points, 
													Total_Particles);
	gpuError_Check( cudaDeviceSynchronize() );


// 2.- RBF interpolation into the fixed grid<DIMENSIONS, TYPE>
	
	// 2.1. - Find near particles
	const UINT	 MaxNeighborNum = fmin(pow(2 * round(DISC_RADIUS) + 1, DIMENSIONS), Adapt_Points);
	const float  search_radius 	= DISC_RADIUS * Base_Mesh.Discr_length();						// max radius to search ([4,6] appears to be optimal)

	const INT	 max_steps 		= 1000;		 		// max steps at the Conjugate Gradient (CG) algorithm
	const float in_tolerance 	= TOLERANCE_ConjGrad; 	// CG stop tolerance
	int16_t err = 0;

	thrust::device_vector<INT>		GPU_Index_array(MaxNeighborNum * Total_Particles,-1);
	thrust::device_vector<float>	GPU_Mat_entries(MaxNeighborNum * Total_Particles, 0);
	thrust::device_vector<UINT>		GPU_Num_Neighbors(Total_Particles, 1);

	if (Adapt_Points < ptSEARCH_THRESHOLD) {
		Exh_PP_Search<DIMENSIONS, TYPE> << <Blocks, Threads >> > (	rpc(Particle_Positions, 0),
														rpc(Particle_Positions, 0),
														rpc(GPU_Index_array, 0),
														rpc(GPU_Mat_entries, 0),
														rpc(GPU_Num_Neighbors, 0),
														MaxNeighborNum,
														Adapt_Points,
														Total_Particles,
														search_radius,
														Base_Mesh);
		gpuError_Check(cudaDeviceSynchronize());
	}
	else {
		err = _CS_Neighbor_Search<DIMENSIONS, TYPE>(Particle_Positions,
										PDF_Particles,
										GPU_Index_array,
										GPU_Mat_entries,
										GPU_Num_Neighbors,
										Adapt_Points,
										MaxNeighborNum,
										search_radius,
										Base_Mesh);

		if (err == -1) { return -1; }
	}

	// Before going to the next step, define the bounding box of the advected particles!
	thrust::device_vector<T> projection(Block_Particles);
	for (uint16_t d = 0; d < DIMENSIONS; d++) {
		findProjection<DIM, T> << <Blocks, Threads >> > (rpc(GPU_Part_Position, 0), rpc(projection, 0), Block_Particles, d);

		T temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end()));
		T temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));

		Supp_BBox.Boundary_inf.dim[d] = fmax(Base_Mesh.Boundary_inf.dim[d], fmin(Supp_BBox.Boundary_inf.dim[d], temp_1));
		Supp_BBox.Boundary_sup.dim[d] = fmin(Base_Mesh.Boundary_sup.dim[d], fmax(Supp_BBox.Boundary_sup.dim[d], temp_2));
	}

	// Final, updated support bounding box?
	Supp_BBox.Boundary_inf = Base_Mesh.Get_node(Base_Mesh.Get_binIdx(Supp_BBox.Boundary_inf, -roundf(DISC_RADIUS)));
	Supp_BBox.Boundary_sup = Base_Mesh.Get_node(Base_Mesh.Get_binIdx(Supp_BBox.Boundary_sup, roundf(DISC_RADIUS)));

	// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
	thrust::device_vector<float>	GPU_lambdas(Total_Particles);	// solution vector (RBF weights)
	thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0.00f);		// this will serve as the initial condition

	err = CONJUGATE_GRADIENT_SOLVE<float>( GPU_lambdas, 
												GPU_Index_array, 
												GPU_Mat_entries, 
												GPU_Num_Neighbors, 
												PDF_Particles, 
												Total_Particles, 
												MaxNeighborNum, 
												max_steps, 
												in_tolerance);
	if (err == -1) { return err; }

// 
// 
// 3.- Reinitialization
	GPU_PDF.resize(Base_Mesh.Total_Nodes(), 0);

for (UINT s = 0; s < Random_Samples; s++){
	Threads = fminf(THREADS_P_BLK, Adapt_Points);
	Blocks = floorf((Adapt_Points - 1) / Threads) + 1;				// To compute the interpolation results at the GPU

	RESTART_GRID_FIND_GN<DIMENSIONS, TYPE><<< Blocks, Threads >>>(rpc(Particle_Positions, 0),
													rpc(GPU_PDF, 0),
													rpc(GPU_lambdas, 0),
													rpc(Impulses, 0),
													search_radius,
													Adapt_Points,
													s,
													Base_Mesh);
	gpuError_Check(cudaDeviceSynchronize());
}			

// Correction of any possible negative PDF values
		// Re-define Threads and Blocks
		Threads = fminf(THREADS_P_BLK, Base_Mesh.Total_Nodes());
		Blocks  = floorf((Base_Mesh.Total_Nodes() - 1) / Threads) + 1;
		
	CORRECTION <TYPE> <<<Blocks, Threads>>>(rpc(GPU_PDF,0), Base_Mesh.Total_Nodes());
	gpuError_Check(cudaDeviceSynchronize());
	

	//std::cout << "Transformation reinitialization: done\n";
	std::cout << "/-------------------------------------------------------------------/\n";

	thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.00f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)
	*PDF = GPU_PDF;

	return 0; // if this works
}

// This function is for the Delta-impulsive case!
int16_t RANDOMIZE_II(	const INT* 						n_samples, 
					const INT 						Total_Samples, 
					std::vector<Impulse_Param_vec>* Parameter_Mesh, 
					const Distributions* 			Dist_Parameters) {

	std::vector<Param_pair> aux_PM;

	for (uint16_t d = 0; d < DIMENSIONS; d++){
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];

		INT err_check = PARAMETER_VEC_BUILD(n_samples[d], PP, Dist_Parameters[d]);
		if (err_check == -1){ return -1; }

		// append to the output array
		aux_PM.insert(aux_PM.end(), &PP[0], &PP[n_samples[d]]);
		delete[] PP;
	}

	for (UINT k = 0; k < Total_Samples; k++){
		// 1st, find the parameter components
		INT aux_num 	=  n_samples[0];
		INT aux_num_2 	=  n_samples[0];

		INT aux_idx = positive_rem(k, aux_num);

		Parameter_Mesh->at(k).sample_vec[0] = aux_PM[aux_idx].sample;
		Parameter_Mesh->at(k).Joint_PDF 	= aux_PM[aux_idx].PDF;

		for (uint16_t d = 1; d < DIMENSIONS; d++){

			aux_idx = floor(positive_rem(k, aux_num * n_samples[d]) / aux_num);

			Parameter_Mesh->at(k).sample_vec[d]  = aux_PM[aux_idx + aux_num_2].sample;
			Parameter_Mesh->at(k).Joint_PDF 	*= aux_PM[aux_idx + aux_num_2].PDF;

			aux_num *= n_samples[d];
			aux_num_2 += n_samples[d];
		}

	}
	return 0;
}
#endif
#endif