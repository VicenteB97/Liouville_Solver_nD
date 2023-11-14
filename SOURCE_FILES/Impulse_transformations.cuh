
//#if (IMPULSE_TYPE == 1) 	// this handles the examples for delta-like impulses
#ifndef __IMPULSE_TRANSFORMATIONS_CUH__
#define __IMPULSE_TRANSFORMATIONS_CUH__

#if IMPULSE_TYPE == 1
#include "Classes.cuh"
#include "Interpolation.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

template<uint16_t PHASE_SPACE_DIM>
int16_t RANDOMIZE_II(	const INT* 					n_samples, 
						const INT 						totalSampleCount, 
						std::vector<Param_vec<PHASE_SPACE_DIM, float>>* Parameter_Mesh, 
						const Distributions* 			Dist_Parameters);


template<uint16_t PHASE_SPACE_DIM, class T>
__global__ void TRANSFORM_PARTICLES(gridPoint<PHASE_SPACE_DIM, T>*					Particle_Locations,
									const Param_vec<PHASE_SPACE_DIM, float>*	impulse_strengths,
									const INT					Num_Particles_per_sample,
									const INT					Total_Particles) {

	const INT i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles){

		const INT j =(INT) floorf((float)i / Num_Particles_per_sample);				// current sample
		Param_vec<PHASE_SPACE_DIM, float> aux = impulse_strengths[j];

		 
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++){			
			Particle_Locations[i].dim[d] += aux.sample_vec[d];		// we only have to add to the second variable!
		}
	}
}


template<uint16_t PHASE_SPACE_DIM, class T>
int16_t IMPULSE_TRANSFORM_PDF(thrust::device_vector<T>&								GPU_PDF,				// PDF in Mesh
							const std::vector<gridPoint<PHASE_SPACE_DIM, class T>>& Particle_Localtions,	// Particle positions
							const std::vector<T>&									Particle_Values,		// PDF in AMR-selected points
							const Time_Impulse_vec									time,					// time-impulse information 
							const INT												jumpCount,				// current jumpCount 
							const grid<PHASE_SPACE_DIM, T>&							Problem_Domain,
							const grid<PHASE_SPACE_DIM, T>&							Expanded_Domain,
							grid<PHASE_SPACE_DIM, T>&								Supp_BBox){	 

// 0.- Create the impulse samples

		UINT Adapt_Points = Particle_Values.size();

		Distributions* Imp_Param_Dist = new Distributions[PHASE_SPACE_DIM];
		INT n_samples[PHASE_SPACE_DIM];

		 
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++){
				// RV mean and variance
				Imp_Param_Dist[d].Name  			= deltaImpulse_distribution_NAMES[jumpCount * PHASE_SPACE_DIM + d];			// N, G or U distributions
				Imp_Param_Dist[d].isTruncated  		= deltaImpulse_distribution_isTRUNC[jumpCount * PHASE_SPACE_DIM + d];		// TRUNCATED?
				Imp_Param_Dist[d].trunc_interval[0] = deltaImpulse_distribution_InfTVAL[jumpCount * PHASE_SPACE_DIM + d];		// min of trunc. interval (if chosen low enough, automatically bounds to -6 std. deviations)
				Imp_Param_Dist[d].trunc_interval[1] = deltaImpulse_distribution_SupTVAL[jumpCount * PHASE_SPACE_DIM + d]; 		// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
				Imp_Param_Dist[d].params[0] 		= deltaImpulse_distribution_MEAN[jumpCount * PHASE_SPACE_DIM + d]; 			// mean
				Imp_Param_Dist[d].params[1] 		= deltaImpulse_distribution_STD[jumpCount * PHASE_SPACE_DIM + d];			// std
				n_samples[d]						= deltaImpulse_distribution_SAMPLES[jumpCount * PHASE_SPACE_DIM + d];		// no. of samples
		}

		INT Random_Samples = 1;
		 
		for (UINT i = 0; i < PHASE_SPACE_DIM; i++){
			Random_Samples *= n_samples[i];
		}

		std::vector<Param_vec<PHASE_SPACE_DIM, T>> Impulse_Parameter_Mesh(Random_Samples);

		INT error_check = RANDOMIZE_II<PHASE_SPACE_DIM>(n_samples, Random_Samples, &Impulse_Parameter_Mesh, Imp_Param_Dist);
		if (error_check == -1){return -1;}

		float Sum_Rand_Params = 0;
		for (INT i = 0; i < Random_Samples; i++) {
			Sum_Rand_Params += Impulse_Parameter_Mesh[i].Joint_PDF;
		}

		std::vector<gridPoint<PHASE_SPACE_DIM, T>>	Full_Adapt_Grid(0);
		std::vector<float>							Full_Particle_Values(0);

		for (INT i = 0; i < Random_Samples; i++) {
			Full_Adapt_Grid		.insert(Full_Adapt_Grid.end(), Adapt_MESH.begin(), Adapt_MESH.end());
			Full_Particle_Values.insert(Full_Particle_Values.end(), Particle_Values.begin(), Particle_Values.end());
		}

		const INT Total_Particles = Full_Particle_Values.size();

// 1.- Do the transformation of the points according to the info given by the time-impulse vector
		thrust::device_vector<gridPoint<PHASE_SPACE_DIM, T>>	Particle_Locations 	= Full_Adapt_Grid;		// To compute the transformations at the GPU
		thrust::device_vector<float>							PDF_Particles	   	= Full_Particle_Values;		// PDF of the transformations at the GPU
		thrust::device_vector<Param_vec<PHASE_SPACE_DIM, T>>	Impulses 			= Impulse_Parameter_Mesh;

		INT Threads = (INT)fmin(THREADS_P_BLK, Total_Particles);
		INT Blocks  = (INT)floor(Total_Particles / Threads) + 1;

	// FOR SOME REASON, I'M ONLY WRITING IN SOME VALUES...NOT ALL OF THEM

	TRANSFORM_PARTICLES <PHASE_SPACE_DIM,T> << <Blocks, Threads >> > (	rpc(Particle_Locations,0), 
																		rpc(Impulses,0), 
																		Adapt_Points, 
																		Total_Particles);
	gpuError_Check( cudaDeviceSynchronize() );


// 3.- Reinitialization
	GPU_PDF.resize(Problem_Domain.Total_Nodes(), 0);

for (UINT s = 0; s < Random_Samples; s++){
	Threads = fmin(THREADS_P_BLK, Adapt_Points);
	Blocks = floor((Adapt_Points - 1) / Threads) + 1;		// To compute the interpolation results at the GPU

	RESTART_GRID_FIND_GN<PHASE_SPACE_DIM, T><<< Blocks, Threads >>>(rpc(Particle_Locations, 0),
																	rpc(GPU_PDF, 0),
																	rpc(GPU_lambdas, 0),
																	rpc(Impulses, 0),
																	search_radius,
																	Adapt_Points,
																	s,
																	Problem_Domain,
																	Expanded_Domain);
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
template<uint16_t PHASE_SPACE_DIM>
int16_t RANDOMIZE_II(const INT* 								n_samples, 
					const INT 									totalSampleCount, 
					std::vector<Param_vec<PHASE_SPACE_DIM, T>>* Parameter_Mesh,
					const Distributions* 						Dist_Parameters) {

	std::vector<Param_pair> aux_PM;

	for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++){
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];

		INT err_check = PARAMETER_VEC_BUILD(n_samples[d], PP, Dist_Parameters[d]);
		if (err_check == -1){ return -1; }

		// append to the output array
		aux_PM.insert(aux_PM.end(), &PP[0], &PP[n_samples[d]]);
		delete[] PP;
	}

	for (UINT k = 0; k < totalSampleCount; k++){
		// 1st, find the parameter components
		INT aux_num 	=  n_samples[0];
		INT aux_num_2 	=  n_samples[0];

		INT aux_idx = positive_rem(k, aux_num);

		Parameter_Mesh->at(k).sample_vec[0] = aux_PM[aux_idx].sample;
		Parameter_Mesh->at(k).Joint_PDF 	= aux_PM[aux_idx].PDF;

		for (uint16_t d = 1; d < PHASE_SPACE_DIM; d++){

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