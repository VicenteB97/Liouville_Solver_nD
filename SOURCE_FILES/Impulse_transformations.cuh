
//#if (IMPULSE_TYPE == 1) 	// this handles the examples for delta-like impulses
#ifndef __IMPULSE_TRANSFORMATIONS_CUH__
#define __IMPULSE_TRANSFORMATIONS_CUH__

#if IMPULSE_TYPE == 1
#include "Classes.cuh"
#include "Mat_Ops.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

int RANDOMIZE_II(const int* n_samples,
	const int 						Total_Samples,
	std::vector<Impulse_Param_vec>* Parameter_Mesh,
	const Distributions* Dist_Parameters);



/// <summary>
/// 
/// </summary>
/// <param name="Particle_Positions"></param>
/// <param name="impulse_strengths"></param>
/// <param name="Num_Particles_per_sample"></param>
/// <param name="Total_Particles"></param>
/// <returns></returns>
__global__ void TRANSFORM_PARTICLES(gridPoint* Particle_Positions,
	const Impulse_Param_vec* impulse_strengths,
	const int					Num_Particles_per_sample,
	const int					Total_Particles) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles) {

		const int j = (int)floorf((float)i / Num_Particles_per_sample);				// current sample
		Impulse_Param_vec aux = impulse_strengths[j];

#pragma unroll
		for (unsigned d = 0; d < DIMENSIONS; d++) {
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
int IMPULSE_TRANSFORM_PDF(const gridPoint* MESH,			// Fixed Mesh
	const std::vector<gridPoint>* Adapt_MESH,		// AMR-selected points
	thrust::host_vector<float>* PDF,			// PDF in Mesh
	const std::vector<float>* Adapt_PDF,		// PDF in AMR-selected points
	const Time_Impulse_vec			time,			// time-impulse information 
	const int						jump,			// current jump 
	const int						Grid_Nodes,		// Number of grid points
	const int 						PtsPerDim,
	thrust::device_vector<gridPoint>& D__Boundary) {

	// 0.- Create the impulse samples

	int Adapt_Points = Adapt_PDF->size();

	Distributions* Imp_Param_Dist = new Distributions[DIMENSIONS];
	int n_samples[DIMENSIONS];

#pragma unroll
	for (unsigned d = 0; d < DIMENSIONS; d++) {
		// RV mean and variance
		Imp_Param_Dist[d].Name = D_JUMP_DIST_NAMES[jump * DIMENSIONS + d];						// N, G or U distributions
		Imp_Param_Dist[d].Truncated = D_JUMP_DIST_TRUNC[jump * DIMENSIONS + d];						// TRUNCATED?
		Imp_Param_Dist[d].trunc_interval[0] = D_JUMP_DIST_InfTVAL[jump * DIMENSIONS + d];						// min of trunc. interval (if chosen low enough, automatically bounds to -6 std. deviations)
		Imp_Param_Dist[d].trunc_interval[1] = D_JUMP_DIST_SupTVAL[jump * DIMENSIONS + d]; 					// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
		Imp_Param_Dist[d].params[0] = D_JUMP_DIST_MEAN[jump * DIMENSIONS + d]; 	// mean
		Imp_Param_Dist[d].params[1] = D_JUMP_DIST_STD[jump * DIMENSIONS + d];		// std
		n_samples[d] = D_JUMP_DIST_SAMPLES[jump * DIMENSIONS + d];			// no. of samples
	}

	int Random_Samples = 1;
#pragma unroll
	for (unsigned i = 0; i < DIMENSIONS; i++) {
		Random_Samples *= n_samples[i];
	}

	std::vector<Impulse_Param_vec> Impulse_Parameter_Mesh(Random_Samples);

	int error_check = RANDOMIZE_II(n_samples, Random_Samples, &Impulse_Parameter_Mesh, Imp_Param_Dist);
	if (error_check == -1) { return -1; }

	float Sum_Rand_Params = 0;
	for (int i = 0; i < Random_Samples; i++) {
		Sum_Rand_Params += Impulse_Parameter_Mesh[i].Joint_PDF;
	}

	std::vector<gridPoint>	Full_Adapt_Grid(0);
	std::vector<float>		Full_Adapt_PDF(0);

	for (int i = 0; i < Random_Samples; i++) {
		Full_Adapt_Grid.insert(Full_Adapt_Grid.end(), Adapt_MESH->begin(), Adapt_MESH->end());
		Full_Adapt_PDF.insert(Full_Adapt_PDF.end(), Adapt_PDF->begin(), Adapt_PDF->end());
	}

	const int Total_Particles = Full_Adapt_PDF.size();

	// 1.- Do the transformation of the points according to the info given by the time-impulse vector
	thrust::device_vector<gridPoint>			Particle_Positions = Full_Adapt_Grid;		// To compute the transformations at the GPU
	thrust::device_vector<float>				PDF_Particles = Full_Adapt_PDF;		// PDF of the transformations at the GPU
	thrust::device_vector<Impulse_Param_vec>	Impulses = Impulse_Parameter_Mesh;

	int Threads = (int)fminf(THREADS_P_BLK, Total_Particles);
	int Blocks = (int)floorf(Total_Particles / Threads) + 1;

	// FOR SOME REASON, I'M ONLY WRITING IN SOME VALUES...NOT ALL OF THEM

	TRANSFORM_PARTICLES << <Blocks, Threads >> > (raw_pointer_cast(&Particle_Positions[0]),
		raw_pointer_cast(&Impulses[0]),
		Adapt_Points,
		Total_Particles);
	gpuError_Check(cudaDeviceSynchronize());


	// 2.- RBF interpolation into the fixed grid

		// 2.1. - Find near particles
	const int	 MaxNeighborNum = fmin(pow(2 * DISC_RADIUS, DIMENSIONS), Adapt_Points);
	const float  disc_X = (MESH[1].dim[0] - MESH[0].dim[0]);	// H_Mesh discretization size (per dimension)
	const float  search_radius = DISC_RADIUS * disc_X;						// max radius to search ([4,6] appears to be optimal)

	const int	 max_steps = 1000;		 		// max steps at the Conjugate Gradient (CG) algorithm
	const float in_tolerance = TOLERANCE_ConjGrad; 	// CG stop tolerance

	thrust::device_vector<int32_t>		GPU_Index_array;
	thrust::device_vector<float>		GPU_Mat_entries;
	thrust::device_vector<uint32_t>		GPU_Num_Neighbors;

	GPU_Index_array.resize(MaxNeighborNum * Total_Particles);
	GPU_Mat_entries.resize(MaxNeighborNum * Total_Particles);
	GPU_Num_Neighbors.resize(Total_Particles);

	Exh_PP_Search << <Blocks, Threads >> > (raw_pointer_cast(&Particle_Positions[0]),
		raw_pointer_cast(&Particle_Positions[0]),
		raw_pointer_cast(&GPU_Index_array[0]),
		raw_pointer_cast(&GPU_Mat_entries[0]),
		raw_pointer_cast(&GPU_Num_Neighbors[0]),
		MaxNeighborNum,
		Adapt_Points,
		Total_Particles,
		search_radius,
		rpc(D__Boundary, 0));
	gpuError_Check(cudaDeviceSynchronize());
	//std::cout << "Transformation Point Search: done\n";

	// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
	thrust::device_vector<float>	GPU_lambdas(Total_Particles);	// solution vector (RBF weights)
	thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0.00f);		// this will serve as the initial condition

	int err = CONJUGATE_GRADIENT_SOLVE<float>(GPU_lambdas,
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

	thrust::device_vector<float> GPU_PDF(Grid_Nodes);
	thrust::fill(GPU_PDF.begin(), GPU_PDF.end(), 0.00f);

	for (unsigned s = 0; s < Random_Samples; s++) {
		Threads = fminf(THREADS_P_BLK, Adapt_Points * Random_Samples);
		Blocks = floorf((Adapt_Points * Random_Samples - 1) / Threads) + 1;				// To compute the interpolation results at the GPU

		RESTART_GRID_FIND_GN_II<TYPE> << < Blocks, Threads >> > (raw_pointer_cast(&Particle_Positions[0]),
			raw_pointer_cast(&GPU_PDF[0]),
			raw_pointer_cast(&GPU_lambdas[0]),
			raw_pointer_cast(&Impulses[0]),
			search_radius,
			MESH[0],
			disc_X,
			PtsPerDim,
			Adapt_Points,
			Random_Samples,
			rpc(D__Boundary, 0));
		gpuError_Check(cudaDeviceSynchronize());
	}

	// Correction of any possible negative PDF values
			// Re-define Threads and Blocks
	Threads = fminf(THREADS_P_BLK, Grid_Nodes);
	Blocks = floorf((Grid_Nodes - 1) / Threads) + 1;

	CORRECTION<TYPE> << <Blocks, Threads >> > (raw_pointer_cast(&GPU_PDF[0]), Grid_Nodes);
	gpuError_Check(cudaDeviceSynchronize());


	//std::cout << "Transformation reinitialization: done\n";
	std::cout << "/-------------------------------------------------------------------/\n";

	thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1.00f / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)

	thrust::copy(GPU_PDF.begin(), GPU_PDF.end(), PDF->begin());

	return 0; // if this works
}

int RANDOMIZE_II(const int* n_samples,
	const int 						Total_Samples,
	std::vector<Impulse_Param_vec>* Parameter_Mesh,
	const Distributions* Dist_Parameters) {

	std::vector<Param_pair> aux_PM;

	for (unsigned d = 0; d < DIMENSIONS; d++) {
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];

		int err_check = PARAMETER_VEC_BUILD(n_samples[d], PP, Dist_Parameters[d]);
		if (err_check == -1) { return -1; }

		// append to the output array
		aux_PM.insert(aux_PM.end(), &PP[0], &PP[n_samples[d]]);
		delete[] PP;
	}

	for (unsigned k = 0; k < Total_Samples; k++) {
		// 1st, find the parameter components
		int aux_num = n_samples[0];
		int aux_num_2 = n_samples[0];

		int aux_idx = positive_rem(k, aux_num);

		Parameter_Mesh->at(k).sample_vec[0] = aux_PM[aux_idx].sample;
		Parameter_Mesh->at(k).Joint_PDF = aux_PM[aux_idx].PDF;

		for (unsigned d = 1; d < DIMENSIONS; d++) {

			aux_idx = floor(positive_rem(k, aux_num * n_samples[d]) / aux_num);

			Parameter_Mesh->at(k).sample_vec[d] = aux_PM[aux_idx + aux_num_2].sample;
			Parameter_Mesh->at(k).Joint_PDF *= aux_PM[aux_idx + aux_num_2].PDF;

			aux_num *= n_samples[d];
			aux_num_2 += n_samples[d];
		}

	}
	return 0;
}
#endif
#endif