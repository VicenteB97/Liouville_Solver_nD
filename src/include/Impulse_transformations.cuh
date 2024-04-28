
//#if (IMPULSE_TYPE == 1) 	// this handles the examples for delta-like impulses
#ifndef __IMPULSE_TRANSFORMATIONS_CUH__
#define __IMPULSE_TRANSFORMATIONS_CUH__

#if IMPULSE_TYPE == 1
#include "headers.cuh"
#include "utils/numeric_defs.cuh"
#include "utils/others.cuh"
#include "Probability.cuh"
#include "Interpolation.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

int16_t RANDOMIZE_II(	const intType* 					n_samples, 
						const intType 						totalSampleCount, 
						std::vector<Param_vec<PHASE_SPACE_DIMENSIONS>>* Parameter_Mesh, 
						const Distributions* 			Dist_Parameters);


__global__ void TRANSFORM_PARTICLES(Particle*					Particle_Locations,
									const Param_vec<PHASE_SPACE_DIMENSIONS>*	impulse_strengths,
									const intType					Num_Particles_per_sample,
									const intType					Total_Particles) {

	const intType i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles){

		const intType j =(intType) floorf((float)i / Num_Particles_per_sample);		// current sample
		Param_vec<PHASE_SPACE_DIMENSIONS> aux = impulse_strengths[j];

		 
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++){			
			Particle_Locations[i].dim[d] += aux.sample_vec[d];				// we only have to add to the second variable!
		}
	}
}

int16_t IMPULSE_TRANSFORM_PDF(thrust::device_vector<floatType>&		GPU_PDF,				// PDF in Mesh
							thrust::device_vector<Particle>& 	D_Particle_Locations,	// Particle positions
							thrust::device_vector<floatType>&		D_Particle_Values,		// PDF in AMR-selected points
							const Time_instants					time,					// time-impulse information 
							const intType							jumpCount,				// current jumpCount 
							const Mesh&							Problem_Domain,
							const Mesh&							Expanded_Domain,
							Mesh&								Supp_BBox){	 

// 0.- Create the impulse samples

		uintType AMR_ActiveNodeCount = D_Particle_Locations.size();

		Distributions* Imp_Param_Dist = new Distributions[PHASE_SPACE_DIMENSIONS];
		intType n_samples[PHASE_SPACE_DIMENSIONS];

		 
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++){
				// RV mean and variance
				Imp_Param_Dist[d].Name  			= deltaImpulse_distribution_NAMES[jumpCount * PHASE_SPACE_DIMENSIONS + d];			// N, G or U distributions
				Imp_Param_Dist[d].isTruncated  		= deltaImpulse_distribution_isTRUNC[jumpCount * PHASE_SPACE_DIMENSIONS + d];		// TRUNCATED?
				Imp_Param_Dist[d].trunc_interval[0] = deltaImpulse_distribution_InfTVAL[jumpCount * PHASE_SPACE_DIMENSIONS + d];		// min of trunc. interval (if chosen low enough, automatically bounds to -6 std. deviations)
				Imp_Param_Dist[d].trunc_interval[1] = deltaImpulse_distribution_SupTVAL[jumpCount * PHASE_SPACE_DIMENSIONS + d]; 		// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
				Imp_Param_Dist[d].params[0] 		= deltaImpulse_distribution_MEAN[jumpCount * PHASE_SPACE_DIMENSIONS + d]; 			// mean
				Imp_Param_Dist[d].params[1] 		= deltaImpulse_distribution_STD[jumpCount * PHASE_SPACE_DIMENSIONS + d];			// std
				n_samples[d]						= deltaImpulse_distribution_SAMPLES[jumpCount * PHASE_SPACE_DIMENSIONS + d];		// no. of samples
		}

		intType Random_Samples = 1;
		 
		for (uintType i = 0; i < PHASE_SPACE_DIMENSIONS; i++){
			Random_Samples *= n_samples[i];
		}

		std::vector<Param_vec<PHASE_SPACE_DIMENSIONS>> Impulse_Parameter_Mesh(Random_Samples);

		errorCheck(RANDOMIZE_II(n_samples, Random_Samples, &Impulse_Parameter_Mesh, Imp_Param_Dist))

		float Sum_Rand_Params = 0;
		for (intType i = 0; i < Random_Samples; i++) {
			Sum_Rand_Params += Impulse_Parameter_Mesh[i].Joint_PDF;
		}		

		const intType Total_Particles = AMR_ActiveNodeCount * Random_Samples;

		D_Particle_Locations.resize(Total_Particles);
		D_Particle_Values.resize(Total_Particles);

		for(uint16_t k = 1; k < Random_Samples; k++){
			thrust::copy(thrust::device, &D_Particle_Locations[0], &D_Particle_Locations[AMR_ActiveNodeCount], &D_Particle_Locations[k * AMR_ActiveNodeCount]);
			thrust::copy(thrust::device, &D_Particle_Values[0], &D_Particle_Values[AMR_ActiveNodeCount], &D_Particle_Values[k * AMR_ActiveNodeCount]);
		}

// 1.- Do the transformation of the points according to the info given by the time-impulse vector
		thrust::device_vector<Param_vec<PHASE_SPACE_DIMENSIONS>> 	Impulses = Impulse_Parameter_Mesh;

		intType Threads = (intType)fmin(THREADS_P_BLK, Total_Particles);
		intType Blocks  = (intType)floor(Total_Particles / Threads) + 1;

	// FOR SOME REASON, I'M ONLY WRITING IN SOME VALUES...NOT ALL OF THEM

	TRANSFORM_PARTICLES << <Blocks, Threads >> > (	rpc(D_Particle_Locations,0), 
													rpc(Impulses,0), 
													AMR_ActiveNodeCount, 
													Total_Particles);
	gpuError_Check( cudaDeviceSynchronize() );


// 3.- Reinitialization
	#if ERASE_dPDF
	GPU_PDF.resize(Problem_Domain.Total_Nodes(), 0);	// PDF is reset to 0, so that we may use atomic adding at the remeshing step
	#else
	thrust::fill(thrust::device, GPU_PDF.begin(), GPU_PDF.end(), 0);
	#endif

	floatType search_radius = DISC_RADIUS * Problem_Domain.Discr_length();

for (uintType s = 0; s < Random_Samples; s++){
	Threads = fmin(THREADS_P_BLK, AMR_ActiveNodeCount);
	Blocks = floor((AMR_ActiveNodeCount - 1) / Threads) + 1;		// To compute the interpolation results at the GPU

	RESTART_GRID_FIND_GN<<< Blocks, Threads >>>(rpc(D_Particle_Locations, 0),
												rpc(GPU_PDF, 0),
												rpc(D_Particle_Values, 0),
												rpc(Impulses, 0),
												search_radius,
												AMR_ActiveNodeCount,
												s,
												Problem_Domain,
												Expanded_Domain);
	gpuError_Check(cudaDeviceSynchronize());
}			

// Correction of any possible negative PDF values
		// Re-define Threads and Blocks
		Threads = fminf(THREADS_P_BLK, Problem_Domain.Total_Nodes());
		Blocks  = floorf((Problem_Domain.Total_Nodes() - 1) / Threads) + 1;
		
		CORRECTION <<<Blocks, Threads>>>(rpc(GPU_PDF,0), Problem_Domain.Total_Nodes());
		gpuError_Check(cudaDeviceSynchronize());
	

	//std::cout << "Transformation reinitialization: done\n";
	std::cout << "/-------------------------------------------------------------------/\n";

	thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.00f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)

	return 0; // if this works
}

// This function is for the Delta-impulsive case!
int16_t RANDOMIZE_II(const intType* 								n_samples, 
					const intType 									totalSampleCount, 
					std::vector<Param_vec<PHASE_SPACE_DIMENSIONS>>* Parameter_Mesh,
					const Distributions* 						Dist_Parameters) {

	std::vector<Param_pair> aux_PM;

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++){
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];

		intType err_check = PARAMETER_VEC_BUILD(n_samples[d], PP, Dist_Parameters[d]);
		if (err_check == -1){ return -1; }

		// append to the output array
		aux_PM.insert(aux_PM.end(), &PP[0], &PP[n_samples[d]]);
		delete[] PP;
	}

	for (uintType k = 0; k < totalSampleCount; k++){
		// 1st, find the parameter components
		intType aux_num 	=  n_samples[0];
		intType aux_num_2 	=  n_samples[0];

		intType aux_idx = positive_rem(k, aux_num);

		Parameter_Mesh->at(k).sample_vec[0] = aux_PM[aux_idx].sample;
		Parameter_Mesh->at(k).Joint_PDF 	= aux_PM[aux_idx].PDF;

		for (uint16_t d = 1; d < PHASE_SPACE_DIMENSIONS; d++){

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