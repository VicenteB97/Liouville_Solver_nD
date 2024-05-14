#pragma once
# if IMPULSE_TYPE == 1

#include "headers.cuh"
#include "utils/numeric_defs.cuh"
#include "utils/others.cuh"
#include "Probability.cuh"
#include "Interpolation.cuh"
#include "Simulation_parameters.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

__global__ void TRANSFORM_PARTICLES(Particle*					Particle_Locations,
									const Param_vec<PHASE_SPACE_DIMENSIONS>*	impulse_strengths,
									const intType					Num_Particles_per_sample,
									const intType					Total_Particles);

// This function is for the Delta-impulsive case!
int16_t RANDOMIZE_II(const intType* 								n_samples, 
					const intType 									totalSampleCount, 
					std::vector<Param_vec<PHASE_SPACE_DIMENSIONS>>* Parameter_Mesh,
					const Distributions* 						Dist_Parameters);

int16_t IMPULSE_TRANSFORM_PDF(thrust::device_vector<floatType>&		GPU_PDF,				// PDF in Mesh
		thrust::device_vector<Particle>& 	D_Particle_Locations,	// Particle positions
		thrust::device_vector<floatType>&		D_Particle_Values,		// PDF in AMR-selected points
		const Time_instants					time,					// time-impulse information 
		const intType							jumpCount,				// current jumpCount 
		const Mesh&							Problem_Domain,
		const Mesh&							Expanded_Domain,
		Mesh&								Supp_BBox);

#endif