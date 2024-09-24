#pragma once

#include "include/headers.hpp"

#if IMPULSE_TYPE == 1

#include "utils/numeric_defs.cuh"
#include "utils/others.cuh"
#include "Probability.cuh"
#include "Interpolation.cuh"
#include "Simulation_parameters.cuh"

__global__ void TRANSFORM_PARTICLES(Particle*					Particle_Locations,
									const Param_vec<PHASE_SPACE_DIMENSIONS>*	impulse_strengths,
									const intType					Num_Particles_per_sample,
									const intType					Total_Particles);

// This function is for the Delta-impulsive case!
int16_t RANDOMIZE_II(const intType* 								n_samples, 
					const intType 									total_sample_count, 
					std::vector<Param_vec<PHASE_SPACE_DIMENSIONS>>* parameter_mesh,
					const Distributions* 						Dist_Parameters);

int16_t IMPULSE_TRANSFORM_PDF(deviceUniquePtr<floatType>&		GPU_PDF,				// PDF in cartesianMesh
		deviceUniquePtr<Particle>& 	fullParticleLocations_dvc,	// Particle positions
		deviceUniquePtr<floatType>&		fullParticleValues_dvc,		// PDF in AMR-selected points
		const Time_instants					time,					// time-impulse information 
		const intType							jumpCount,				// current jumpCount 
		const cartesianMesh&							Problem_Domain,
		const cartesianMesh&							Expanded_Domain,
		cartesianMesh&								Supp_BBox);

#endif