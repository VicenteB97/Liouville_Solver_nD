
//#if (IMPULSE_TYPE == 1) 	// this handles the examples for delta-like impulses
#ifndef __IMPULSE_TRANSFORMATIONS_CUH__
#define __IMPULSE_TRANSFORMATIONS_CUH__

#if IMPULSE_TYPE == 1
#include "Classes.cuh"
#include "Interpolation.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

template<uint16_t DIM>
int16_t RANDOMIZE_II(	const INT* 					n_samples, 
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
template<uint16_t DIM, class T>
__global__ void TRANSFORM_PARTICLES(gridPoint<DIM, T>*					Particle_Positions,
									const Impulse_Param_vec*	impulse_strengths, 
									const INT					Num_Particles_per_sample,
									const INT					Total_Particles) {

	const INT i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles){

		const INT j =(INT) floorf((float)i / Num_Particles_per_sample);				// current sample
		Impulse_Param_vec aux = impulse_strengths[j];

		 
		for (uint16_t d = 0; d < DIM; d++){			
			Particle_Positions[i].dim[d] += aux.sample_vec[d];		// we only have to add to the second variable!
		}
	}
}

template<uint16_t DIM, class T>
int16_t IMPULSE_TRANSFORM_PDF(const std::vector<gridPoint<DIM, T>>&	Adapt_MESH,		// AMR-selected points
							thrust::host_vector<T>*		PDF,			// PDF in Mesh
							thrust::device_vector<T>&	GPU_PDF,			// PDF in Mesh
							const std::vector<T>&		Adapt_PDF,		// PDF in AMR-selected points
							const Time_Impulse_vec			time,			// time-impulse information 
							const INT						jump,			// current jump 
							const grid<DIM, T>&	Problem_Domain,
							const grid<DIM, T>&	Underlying_Mesh,
							grid<DIM, T>&			Supp_BBox){	 

// 0.- Create the impulse samples

		UINT Adapt_Points = Adapt_PDF.size();

		Distributions* Imp_Param_Dist = new Distributions[DIM];
		INT n_samples[DIM];

		 
		for (uint16_t d = 0; d < DIM; d++){
				// RV mean and variance
				Imp_Param_Dist[d].Name  			= D_JUMP_DIST_NAMES[jump * DIM + d];						// N, G or U distributions
				Imp_Param_Dist[d].Truncated  		= D_JUMP_DIST_TRUNC[jump * DIM + d];						// TRUNCATED?
				Imp_Param_Dist[d].trunc_interval[0] = D_JUMP_DIST_InfTVAL[jump * DIM + d];					// min of trunc. interval (if chosen low enough, automatically bounds to -6 std. deviations)
				Imp_Param_Dist[d].trunc_interval[1] = D_JUMP_DIST_SupTVAL[jump * DIM + d]; 					// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
				Imp_Param_Dist[d].params[0] 		= D_JUMP_DIST_MEAN[jump * DIM + d]; 	// mean
				Imp_Param_Dist[d].params[1] 		= D_JUMP_DIST_STD[jump * DIM + d];		// std
				n_samples[d]						= D_JUMP_DIST_SAMPLES[jump * DIM + d];			// no. of samples
		}

		INT Random_Samples = 1;
		 
		for (UINT i = 0; i < DIM; i++){
			Random_Samples *= n_samples[i];
		}

		std::vector<Impulse_Param_vec> Impulse_Parameter_Mesh(Random_Samples);

		INT error_check = RANDOMIZE_II<DIM>(n_samples, Random_Samples, &Impulse_Parameter_Mesh, Imp_Param_Dist);
		if (error_check == -1){return -1;}

		float Sum_Rand_Params = 0;
		for (INT i = 0; i < Random_Samples; i++) {
			Sum_Rand_Params += Impulse_Parameter_Mesh[i].Joint_PDF;
		}

		std::vector<gridPoint<DIM, T>>	Full_Adapt_Grid(0);
		std::vector<float>				Full_Adapt_PDF(0);

		for (INT i = 0; i < Random_Samples; i++) {
			Full_Adapt_Grid.insert(Full_Adapt_Grid.end(), Adapt_MESH.begin(), Adapt_MESH.end());
			Full_Adapt_PDF.insert(Full_Adapt_PDF.end(), Adapt_PDF.begin(), Adapt_PDF.end());
		}

		const INT Total_Particles = Full_Adapt_PDF.size();

// 1.- Do the transformation of the points according to the info given by the time-impulse vector
		thrust::device_vector<gridPoint<DIM, T>>	Particle_Positions	= Full_Adapt_Grid;		// To compute the transformations at the GPU
		thrust::device_vector<float>				PDF_Particles		= Full_Adapt_PDF;		// PDF of the transformations at the GPU
		thrust::device_vector<Impulse_Param_vec>	Impulses			= Impulse_Parameter_Mesh;

		INT Threads = (INT)fmin(THREADS_P_BLK, Total_Particles);
		INT Blocks  = (INT)floor(Total_Particles / Threads) + 1;

	// FOR SOME REASON, I'M ONLY WRITING IN SOME VALUES...NOT ALL OF THEM

	TRANSFORM_PARTICLES <DIM,T> << <Blocks, Threads >> > (	rpc(Particle_Positions,0), 
															rpc(Impulses,0), 
															Adapt_Points, 
															Total_Particles);
	gpuError_Check( cudaDeviceSynchronize() );


// 2.- RBF interpolation into the fixed grid<DIM, T>
	
	// 2.1. - Find near particles
	const UINT	 MaxNeighborNum = fmin(pow(2 * round(DISC_RADIUS) + 1, DIM), Adapt_Points);
	const float  search_radius 	= DISC_RADIUS * Problem_Domain.Discr_length();						// max radius to search ([4,6] appears to be optimal)

	const INT	 max_steps 		= 1000;		 		// max steps at the Conjugate Gradient (CG) algorithm
	const float in_tolerance 	= TOLERANCE_ConjGrad; 	// CG stop tolerance
	int16_t err = 0;

	// Before going to the next step, define the bounding box of the advected particles!
	thrust::device_vector<T> projection(Total_Particles);

	for (uint16_t d = 0; d < DIM; d++) {
		findProjection<DIM, T> << <Blocks, Threads >> > (rpc(Particle_Positions, 0), rpc(projection, 0), Total_Particles, d);

		T temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end())); // min element from the projection in that direction
		T temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));

		Supp_BBox.Boundary_inf.dim[d] = fmax(Problem_Domain.Boundary_inf.dim[d], temp_1 - ceil(DISC_RADIUS) * Problem_Domain.Discr_length());
		Supp_BBox.Boundary_sup.dim[d] = fmin(Problem_Domain.Boundary_sup.dim[d], temp_2 + ceil(DISC_RADIUS) * Problem_Domain.Discr_length());
	}


	thrust::device_vector<INT>	GPU_Index_array(MaxNeighborNum * Total_Particles,-1);
	thrust::device_vector<T>	GPU_Mat_entries(MaxNeighborNum * Total_Particles, 0);
	thrust::device_vector<UINT>	GPU_Num_Neighbors(Total_Particles, 1);

	if (Adapt_Points < ptSEARCH_THRESHOLD) {
		Exh_PP_Search<DIM, T> << <Blocks, Threads >> > (rpc(Particle_Positions, 0),
														rpc(Particle_Positions, 0),
														rpc(GPU_Index_array, 0),
														rpc(GPU_Mat_entries, 0),
														rpc(GPU_Num_Neighbors, 0),
														MaxNeighborNum,
														Adapt_Points,
														Total_Particles,
														search_radius);
		gpuError_Check(cudaDeviceSynchronize());
	}
	else {
		err = _CS_Neighbor_Search<DIM, T>(Particle_Positions,
										PDF_Particles,
										GPU_Index_array,
										GPU_Mat_entries,
										GPU_Num_Neighbors,
										Adapt_Points,
										MaxNeighborNum,
										Supp_BBox,
										search_radius);

		if (err == -1) { return -1; }
	}

	// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
	thrust::device_vector<T>	GPU_lambdas(Total_Particles);	// solution vector (RBF weights)
	thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0.00f);		// this will serve as the initial condition

	err = CONJUGATE_GRADIENT_SOLVE<T>( GPU_lambdas, 
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
	GPU_PDF.resize(Problem_Domain.Total_Nodes(), 0);

for (UINT s = 0; s < Random_Samples; s++){
	Threads = fmin(THREADS_P_BLK, Adapt_Points);
	Blocks = floor((Adapt_Points - 1) / Threads) + 1;				// To compute the interpolation results at the GPU

	RESTART_GRID_FIND_GN<DIM, T><<< Blocks, Threads >>>(rpc(Particle_Positions, 0),
													rpc(GPU_PDF, 0),
													rpc(GPU_lambdas, 0),
													rpc(Impulses, 0),
													search_radius,
													Adapt_Points,
													s,
													Problem_Domain,
													Underlying_Mesh);
	gpuError_Check(cudaDeviceSynchronize());
}			

// Correction of any possible negative PDF values
		// Re-define Threads and Blocks
		Threads = fminf(THREADS_P_BLK, Problem_Domain.Total_Nodes());
		Blocks  = floorf((Problem_Domain.Total_Nodes() - 1) / Threads) + 1;
		
	CORRECTION <T> <<<Blocks, Threads>>>(rpc(GPU_PDF,0), Problem_Domain.Total_Nodes());
	gpuError_Check(cudaDeviceSynchronize());
	

	//std::cout << "Transformation reinitialization: done\n";
	std::cout << "/-------------------------------------------------------------------/\n";

	thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.00f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)
	*PDF = GPU_PDF;

	return 0; // if this works
}

// This function is for the Delta-impulsive case!
template<uint16_t DIM>
int16_t RANDOMIZE_II(	const INT* 						n_samples, 
					const INT 						Total_Samples, 
					std::vector<Impulse_Param_vec>* Parameter_Mesh, 
					const Distributions* 			Dist_Parameters) {

	std::vector<Param_pair> aux_PM;

	for (uint16_t d = 0; d < DIM; d++){
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

		for (uint16_t d = 1; d < DIM; d++){

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