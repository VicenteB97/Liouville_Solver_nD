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

#ifndef __MAT_OPS_CUH__
#define __MAT_OPS_CUH__

#include "Constants.cuh"
#include "Probability.cuh"
#include "Domain.cuh"

__device__ inline TYPE RBF(const TYPE& SuppRBF, const TYPE& inputNormalized);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 
/// @param Bin_locations 
/// @param Bin_count 
/// @param Bin_local_accSum 
/// @param Search_Particles 
/// @param Bounding_Box 
/// @param Total_Particles 
/// @return 
__global__ void Bin_Insertion_Count(UINT*		Bin_locations,
									UINT*		Bin_count,
									UINT*		Bin_local_accSum,
									const Particle* Search_Particles,
									const Mesh	Bounding_Box,
									const UINT	Total_Particles){
		
	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= Total_Particles) { return; }

	Particle temp_GP = Search_Particles[globalID];

	// Find the Bin where it belongs. We use an unsigned int because since we are working with the bounding box,
	// we're never going to have a point outside. Therefore, there will be a positive index for the bin location.

	UINT bin_idx = Bounding_Box.Get_binIdx(temp_GP);	// this part should be positive...why aren't we getting a positive number?

	// Now, we've got the nearest bin index. This tells the idx of the bin center containing a given particle
	Bin_locations[globalID] = bin_idx;

	//Add one to the number of particles inside the current bin, and store the previous value!
	Bin_local_accSum[globalID] = atomicAdd(&Bin_count[bin_idx], 1);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 
/// @param fixed_Particles 
/// @param Particles_2sort 
/// @param fixed_values 
/// @param values_2sort 
/// @param Bin_count 
/// @param Bin_locations 
/// @param Bin_local_accSum 
/// @param Total_Particles 
/// @param offset 
/// @return 
__global__ void Count_sort( const Particle*fixed_Particles,
							Particle*		Particles_2sort, 
							const TYPE*				fixed_values, 
							TYPE*						values_2sort,
							const UINT*				Bin_count, 
							const UINT*				Bin_locations,
							const UINT*				Bin_local_accSum,	
							const UINT				Total_Particles,
							const INT				offset) {

	const UINT globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= Total_Particles) { return; }

// read bin location (@ the bounding box)
	UINT my_bin_idx = Bin_locations[globalID];

// read the prefix sum of the bin!
	UINT pref_sum_idx = Bin_count[my_bin_idx];

// read local cumulative sum
	UINT my_count_loc = Bin_local_accSum[globalID];

// Add to ordered particle array (pref. sum tells the first indx of the particles to be added, where local_count)
// tells the position among those particles
	UINT final_idx = pref_sum_idx + my_count_loc;

	Particles_2sort[final_idx]  = fixed_Particles[globalID + offset];
	values_2sort[final_idx] 	= fixed_values[globalID + offset];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 
/// @tparam PHASE_SPACE_DIMENSIONS
/// @tparam TYPE
/// @param Search_Particles 
/// @param Bin_count 
/// @param Index_Array 
/// @param Matrix_Entries 
/// @param Num_Neighbors 
/// @param max_neighbor_num 
/// @param Total_Particles 
/// @param offset 
/// @param search_distance 
/// @param Bounding_Box 
/// @return 
__global__ void Neighbor_search(Particle*		Search_Particles,
								const UINT* Bin_count,
								INT*		Index_Array,
								TYPE*		Matrix_Entries,
								const UINT	max_neighbor_num,
								const UINT	Total_Particles,
								const UINT	offset,
								const TYPE			search_distance,			// This tells us how many discretizations we have to move back to find initial bin to search from
								const Mesh		Bounding_Box) {

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// MAYBE: We could use shared memory to access particles nearby (same/near bins)!
	if (globalID >= Total_Particles) { return; }

// Go through the corresponding bins and their particles! Note that we've got to visit some nodes in every direction
// That includes left and right (1D) + up and down (if 2D) + front and behind (if 3D) + ...
// But! Since we know the particles assigned in each bin and how many particles there are in each bin, we can visit efficiently!
	
// Read the input from the sorted Particle array
	Particle fixed_GP = Search_Particles[globalID + offset];

	UINT temp_counter = 0;
	UINT temp_ID = (globalID + offset) * max_neighbor_num;

// Now, we go to the lowest bin and we search all the way up!
	const UINT Bin_offset = lroundf(DISC_RADIUS);

// We find the lowest corner in the neighboring bins (Hence, the Bin_offset variable)
	UINT bin_idx = Bounding_Box.Get_binIdx(fixed_GP, -Bin_offset);

	for (UINT k = 0; k < pow(2 * Bin_offset + 1, PHASE_SPACE_DIMENSIONS); k++) { // That's the total number of bins to visit

		// First: find the global index of the bin to search!
		INT aux_index = bin_idx;
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			aux_index += floorf(positive_rem(k, (UINT)pow(2 * Bin_offset + 1, d + 1)) / (UINT)pow(2 * Bin_offset + 1, d))
			 				* (UINT)pow(Bounding_Box.Nodes_per_Dim, d);
		}

		if (aux_index > -1 && aux_index < Bounding_Box.Total_Nodes()) { // If we are selecting nodes inside the domain

			// Now that we have the bin to search, all we have to do is go through the particles that are in the corresponding bin:
			UINT startIdx	= Bin_count[aux_index];
			UINT endIdx		= Total_Particles;

			// When we haven't reached the last element of the count vector, we have to know where do we stop the counting
			if (aux_index < Bounding_Box.Total_Nodes() - 1) {
				endIdx = Bin_count[aux_index + 1];
			}

			// Add the points in the current bin
			for (UINT m = startIdx; m < endIdx; m++) {

				Particle	temp_GP = Search_Particles[m + offset];
				TYPE		dist = fixed_GP.Distance(temp_GP) / search_distance;

				if (dist <= 1 && temp_counter < max_neighbor_num) {
					// include particle index into COO-sparse information
					// include RBF-distance value into matrix values
					// add count to number of neighbors (maybe optional)
					Index_Array[temp_ID + temp_counter] = m + offset;
					Matrix_Entries[temp_ID + temp_counter] = RBF(search_distance, dist);
					temp_counter++;
				}
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Counting sort! This operation orders the Mesh indeces and it sets the number of nodes inside each bin

/// @brief 
/// @tparam TYPE 
/// @tparam PHASE_SPACE_DIMENSIONS 
/// @param Search_Particles 
/// @param PDF_vals 
/// @param Index_Array 
/// @param Matrix_Entries 
/// @param Num_Neighbors 
/// @param Adapt_Points 
/// @param max_neighbor_num 
/// @param Bounding_Box 
/// @param search_radius 
/// @return 
__host__ int16_t CS_Neighbor_Search(thrust::device_vector<Particle>& Search_Particles,
									thrust::device_vector<TYPE> &PDF_vals,
									thrust::device_vector<INT>& Index_Array,
									thrust::device_vector<TYPE>& Matrix_Entries,
									const UINT Adapt_Points, 
									const UINT max_neighbor_num,
									const Mesh& Bounding_Box,
									const TYPE search_radius) {

	// We need our particles
	UINT Total_Particles = Search_Particles.size();

	thrust::device_vector<UINT> Bin_locations(Adapt_Points,0);

	thrust::device_vector<Particle> temp_Particles(Adapt_Points);		// this one will store the ordered particles!
	thrust::device_vector<TYPE>				 temp_values(Adapt_Points,0);		// this one will store the values of the particles in the new order!

	// Create a bounding box covering the same area as the bounding box from the particles, but change the number of nodes per dim!
	Mesh CS_BBox(Bounding_Box);
	CS_BBox.Nodes_per_Dim = 64;

	// Bin the particles!
	UINT Threads = fmin(THREADS_P_BLK, Adapt_Points);
	UINT Blocks  = floor((Adapt_Points - 1) / Threads) + 1;

	for (UINT k = 0; k < Total_Particles / Adapt_Points; k++) { // We've got to repeat this once per random sample...we'll see another way out...

		UINT offset = k * Adapt_Points;

	// Reinitialize the particle count array
		thrust::device_vector<UINT> Bin_globalCount(CS_BBox.Total_Nodes(), 0);
		thrust::device_vector<UINT> Bin_localCount(Adapt_Points, 0);

	// Insert and count the particles inside each bin
		Bin_Insertion_Count<< <Threads, Blocks >> > (rpc(Bin_locations, 0), 
													rpc(Bin_globalCount, 0),
													rpc(Bin_localCount, 0),
													rpc(Search_Particles, offset), 
													CS_BBox, 
													Adapt_Points);
		gpuError_Check(cudaDeviceSynchronize());

	// Prefix sum of the bin-count array, which gives the cumulative points in each bin (without including the bin itself; thus the name exclusive)
		thrust::exclusive_scan(thrust::device, Bin_globalCount.begin(), Bin_globalCount.end(), Bin_globalCount.begin());

	// Counting sort...we want to sort Search_Particles!
		Count_sort<< <Threads, Blocks >> > (rpc(Search_Particles, 0), 
											rpc(temp_Particles, 0), 
											rpc(PDF_vals,0),
											rpc(temp_values,0),
											rpc(Bin_globalCount, 0),
											rpc(Bin_locations, 0), 
											rpc(Bin_localCount, 0),
											Adapt_Points,
											offset);
		gpuError_Check(cudaDeviceSynchronize());

	// Relabel the particles and their values
		thrust::copy(temp_Particles.begin(), temp_Particles.end(), &Search_Particles[offset]);
		thrust::copy(temp_values.begin(), temp_values.end(), &PDF_vals[offset]);

	// Find particle neighbors using all the previous information
		Neighbor_search<< <Threads, Blocks >> >(rpc(Search_Particles, 0),
												rpc(Bin_globalCount, 0),
												rpc(Index_Array, 0),
												rpc(Matrix_Entries, 0),
												max_neighbor_num,
												Adapt_Points,
												offset,
												search_radius,
												CS_BBox);
		gpuError_Check(cudaDeviceSynchronize());
	}
	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/// @brief 
/// @tparam PHASE_SPACE_DIMENSIONS
/// @tparam TYPE
/// @param Search_Particles 
/// @param Fixed_Particles 
/// @param Index_Array 
/// @param Matrix_Entries
/// @param max_neighbor_num 
/// @param Adapt_Points 
/// @param Total_Particles 
/// @param search_radius 
/// @return 
__global__ void Exh_PP_Search(const Particle* Search_Particles,
							const Particle* Fixed_Particles,
							INT* Index_Array,
							TYPE* Matrix_Entries,
							const INT max_neighbor_num,
							const UINT Adapt_Points,
							const UINT Total_Particles,
							const TYPE search_radius) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Total_Particles) { return; }

	Particle	FP_aux	= Fixed_Particles[i];									// Tells me what parameter sample I'm at
	const INT	k		= i * max_neighbor_num;

	Index_Array[k]	= i;
	Matrix_Entries[k] = RBF(search_radius, 0);

	UINT		aux = 1;
	const INT	i_aux = (double)floor((double)i / Adapt_Points);	// this double is used so that the compiler knows that we are using the CUDA version
	TYPE			dist;

	for (UINT j = i_aux * Adapt_Points; j < (i_aux + 1) * Adapt_Points; j++) {		// neighborhood where I'm searching

		Particle Temp_Particle = Search_Particles[j];

		dist = Temp_Particle.Distance(FP_aux) / search_radius;	// normalized distance between particles

		if (dist <= 1 && aux < max_neighbor_num && j != i) {
			Index_Array[k + aux] = j;
			Matrix_Entries[k + aux] = RBF(search_radius, dist);
			aux++;
		}
	}
}


int16_t particleNeighborSearch(thrust::device_vector<Particle>& Search_Particles,
								thrust::device_vector<TYPE> &PDF_vals,
								thrust::device_vector<INT>& Index_Array,
								thrust::device_vector<TYPE>& Matrix_Entries,
								const UINT Adapt_Points, 
								const UINT MaxNeighborNum,
								const Mesh& Bounding_Box,
								const TYPE search_radius){
									
	uint16_t	Threads = fmin(THREADS_P_BLK, Adapt_Points);
	UINT		Blocks	= floor((Adapt_Points - 1) / Threads) + 1;

	// Dynamical choice of either exhaustive or counting sort-based point search
	if (Adapt_Points < ptSEARCH_THRESHOLD) {
		Exh_PP_Search<< <Blocks, Threads >> > (rpc(Search_Particles, 0),
											rpc(Search_Particles, 0),
											rpc(Index_Array, 0),
											rpc(Matrix_Entries, 0),
											MaxNeighborNum,
											Adapt_Points,
											Adapt_Points,
											search_radius);
		gpuError_Check(cudaDeviceSynchronize());
	}
	else {
		errorCheck(CS_Neighbor_Search(Search_Particles,
										PDF_vals,
										Index_Array,
										Matrix_Entries,
										Adapt_Points,
										MaxNeighborNum,
										Bounding_Box,
										search_radius));
	}
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void UPDATE_VEC(TYPE* x, const TYPE* x0, const TYPE scalar, const TYPE* v, const INT Max_Length) {
	const uint64_t globalIDX = blockDim.x * blockIdx.x + threadIdx.x;

	uint64_t i = globalIDX * ELEMENTS_AT_A_TIME;

	#pragma unroll
	for(uint16_t k = 0; k < ELEMENTS_AT_A_TIME; k++){
		if ((i + k) < Max_Length) { x[i + k] = x0[i + k] + scalar * v[i + k]; }
		else{ return; }
	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void MATRIX_VECTOR_MULTIPLICATION(TYPE* X, const TYPE* x0, const INT* Matrix_idxs, const TYPE* Matrix_entries, const INT total_length, const INT Max_Neighbors) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;	// For each i, which represents the matrix row, we read the index positions and multiply against the particle weights

	if (i >= total_length) { return; }						// total length = adapt_points * totalSampleCount

// 1.- Compute A*X0										
	// 1.1.- Determine where my particles are!!
	const UINT i0 = i * Max_Neighbors;		// where does my search index start

	TYPE a = 0;					// auxiliary value for sum (the diagonal is always 1 in our case)
	UINT j = i0;
	while(Matrix_idxs[j] != -1){
		INT p = Matrix_idxs[j];

		a += Matrix_entries[j] * x0[p]; 		// < n calls to global memory
		j++;
	}

// 2.- Output
	X[i] = a;								// particle weights
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__host__ INT CONJUGATE_GRADIENT_SOLVE(	thrust::device_vector<TYPE>&		GPU_lambdas,
									thrust::device_vector<INT>& GPU_Index_array,
									thrust::device_vector<TYPE>&		GPU_Mat_entries,
									thrust::device_vector<TYPE>&		GPU_AdaptPDF,
									const INT					Total_Particles,
									const INT					MaxNeighborNum,
									const UINT					max_steps,
									const TYPE							in_tolerance) {


	// Determine threads and blocks for the simulation
	const UINT Threads = (UINT)fminf(THREADS_P_BLK, Total_Particles);
	const UINT Blocks  = (UINT)floorf((Total_Particles - 1) / Threads) + 1;
	
	// These are for the update_vec function
	const UINT Threads_2 = (UINT)fminf(THREADS_P_BLK, (float)Total_Particles / ELEMENTS_AT_A_TIME);
	const UINT Blocks_2  = (UINT)floorf((float)(Total_Particles / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;

	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------------------- //
	thrust::device_vector<TYPE>	GPU_R	(Total_Particles);		// residual vector
	thrust::device_vector<TYPE>	GPU_temp(Total_Particles);		// auxiliary vector for computation storage
	thrust::device_vector<TYPE>	GPU_AP	(Total_Particles);		// vector for storing the A*P multiplication
	thrust::device_vector<TYPE>	GPU_P	(Total_Particles);		// P vector
	// ------------------------------------------------------------------------------------------ //

	// Auxiliary values
	INT  k	= 1;	// to control possible overflow of iterations
	bool flag	= true;	// to stop the iterations

// Initialize Conjugate gradient method ----------------------------------------------------
	// Compute A * X0
	MATRIX_VECTOR_MULTIPLICATION << < Blocks, Threads >> > (rpc(GPU_temp, 0), rpc(GPU_lambdas,0), rpc(GPU_Index_array,0),
		rpc(GPU_Mat_entries,0), Total_Particles, MaxNeighborNum);
	gpuError_Check(cudaDeviceSynchronize());

	// Compute R=B-A*X0
	UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(GPU_R,0), rpc(GPU_AdaptPDF,0), (TYPE)-1, rpc(GPU_temp,0), Total_Particles);
	gpuError_Check(cudaDeviceSynchronize());

	TYPE Alpha, R0_norm, r_norm, aux, beta;

	GPU_P = GPU_R;

	while (flag) { // this flag is useful to know when we have arrived to the desired tolerance
	// Alpha computation (EVERYTHING IS CORRECT!)
		// 1.1.- Compute AP=A*P
		MATRIX_VECTOR_MULTIPLICATION << < Blocks, Threads >> > (rpc(GPU_AP,0), rpc(GPU_P,0), rpc(GPU_Index_array,0),
			rpc(GPU_Mat_entries,0), Total_Particles, MaxNeighborNum);
		gpuError_Check(cudaDeviceSynchronize());

		// 1.2.- Compute P'*AP
		thrust::transform(GPU_P.begin(), GPU_P.end(), GPU_AP.begin(), GPU_temp.begin(), thrust::multiplies<TYPE>());
		aux = thrust::reduce(thrust::device, GPU_temp.begin(), GPU_temp.end());

		// 1.3.- R'*R
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_temp.begin(), thrust::multiplies<TYPE>());
		R0_norm = thrust::reduce(thrust::device, GPU_temp.begin(), GPU_temp.end());

		Alpha = R0_norm / aux;

		// New X and R: (new, old, scalar, driving vec, total length)
		// 1.- Update Lambdas
		UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(GPU_lambdas,0), rpc(GPU_lambdas,0), Alpha, rpc(GPU_P,0), Total_Particles);
		// we DO NOT use cudaDeviceSynchronize() because the following CUDA kernel does not require this kernel to be done...we may save a (very small) amount of time

		// 2.- Update residuals 
		UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(GPU_R,0), rpc(GPU_R,0), -Alpha, rpc(GPU_AP,0), Total_Particles);
		gpuError_Check(cudaDeviceSynchronize());

		// Compute residual l_2 norm
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_temp.begin(), thrust::multiplies<TYPE>());
		r_norm = thrust::reduce(thrust::device, GPU_temp.begin(), GPU_temp.end()); // sum of its elements
		r_norm = sqrt(r_norm);

		if ((TYPE) r_norm / Total_Particles < in_tolerance) {
			flag = false;
			break;
		}
		else if (k > max_steps) {
			std::cout << "No convergence was obtained after reaching max. allowed iterations. Last residual norm was: " << r_norm << "\n";
			std::cout << border_mid;

			k = -1;
			flag = false;
			break;
		}
		else {
			beta = r_norm * r_norm / R0_norm;

			UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(GPU_P,0), rpc(GPU_R,0), beta, rpc(GPU_P,0), Total_Particles);
			gpuError_Check(cudaDeviceSynchronize());
			k++;
		}
	}
	return k;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void RESTART_GRID_FIND_GN(Particle* Particle_Positions,
									TYPE* PDF,
									TYPE* lambdas,
									const Param_pair* Parameter_Mesh,
									const INT* n_Samples,
									const TYPE 	 		search_radius,
									const UINT	 		Adapt_Pts,
									const UINT	 		Block_samples,
									const UINT	 		offset,
									const Mesh 	Domain,
									const Mesh	Expanded_Domain) {
	
	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Adapt_Pts * Block_samples) { return; }

	UINT		Current_sample = offset + floorf(i / Adapt_Pts);
	Param_vec<PARAM_SPACE_DIMENSIONS>	aux = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(Current_sample, Parameter_Mesh, n_Samples);

	TYPE weighted_lambda = lambdas[i] * aux.Joint_PDF;

	Particle	particle = Particle_Positions[i];

	// Find the point in the lowest corner of the search box!
	Particle Lowest_node = Expanded_Domain.Get_node(Expanded_Domain.Get_binIdx(particle, -lround(DISC_RADIUS)));

	const INT Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;

	// Go through all the nodes where rewriting will be possible
	for (uint16_t k = 0; k < pow(Neighbors_per_dim, PHASE_SPACE_DIMENSIONS); k++) {

		Particle visit_node = Lowest_node;

		// Get the node at that point
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT temp_idx = floor(positive_rem(k, pow(Neighbors_per_dim, d + 1)) / pow(Neighbors_per_dim, d));
			visit_node.dim[d] += temp_idx * Domain.Discr_length();
		}

		// If it is inside our problem mesh...
		if (Domain.Contains_particle(visit_node)) {

			// Calculate normalized distance
			TYPE dist = visit_node.Distance(particle) / search_radius;

			// if it's inside the RBF support...
			if (dist <= 1) {

				dist = RBF(search_radius, dist) * weighted_lambda;

				INT idx = Domain.Get_binIdx(visit_node);

				atomicAdd(&PDF[idx], dist);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void RESTART_GRID_FIND_GN(Particle*	Particle_Positions,
									TYPE*		PDF,
									TYPE*		lambdas,
									const Param_vec<PHASE_SPACE_DIMENSIONS>* Impulse_weights,
									const TYPE 	search_radius,
									const UINT	Adapt_Pts,
									const UINT	Current_sample,
									const Mesh	Domain,
									const Mesh	Expanded_Domain) {
	// OUTPUT: New values of the PDF at the fixed Mesh

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Adapt_Pts) { return; }

	TYPE weighted_lambda = lambdas[i + Current_sample * Adapt_Pts] * Impulse_weights[Current_sample].Joint_PDF;				// the specific sample weight

	Particle particle(Particle_Positions[i + Current_sample * Adapt_Pts]);

	// Find the point in the lowest corner of the search box!
	Particle Lowest_node = Expanded_Domain.Get_node(Expanded_Domain.Get_binIdx(particle, -roundf(DISC_RADIUS)));

	const INT Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;

	// Go through all the nodes where rewriting will be possible
	for (uint16_t k = 0; k < pow(Neighbors_per_dim, PHASE_SPACE_DIMENSIONS); k++) {

		Particle visit_node = Lowest_node;

		// Get the node at that point
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT temp_idx = floor(positive_rem(k, pow(Neighbors_per_dim, d + 1)) / pow(Neighbors_per_dim, d));
			visit_node.dim[d] += temp_idx * Domain.Discr_length();
		}

		// If it is inside our problem mesh...
		if (Domain.Contains_particle(visit_node)) {

			// Calculate normalized distance
			TYPE dist = visit_node.Distance(particle) / search_radius;

			// if it's inside the RBF support...
			if (dist <= 1) {

				dist = RBF(search_radius, dist) * weighted_lambda;

				INT idx = Domain.Get_binIdx(visit_node);

				atomicAdd(&PDF[idx], dist);
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief This function makes sure that the PDF does not have any negative values! (Having this enforced by the interpolation would've been wonderful)
/// @param PDF 
/// @param Grid_Nodes 
/// @return 

__global__ void CORRECTION(TYPE* PDF, const INT Grid_Nodes){
	const uint64_t globalIDX = blockDim.x * blockIdx.x + threadIdx.x;

	const uint64_t i = globalIDX * ELEMENTS_AT_A_TIME;

	#pragma unroll
	for(uint16_t k = 0; k < ELEMENTS_AT_A_TIME; k++){
		if (i + k < Grid_Nodes){
			PDF[i + k] = fmaxf(PDF[i + k], 0.00f);
		}
	}

	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ TYPE RBF(const TYPE& SuppRBF, const TYPE& inputNormalized){

	#if PHASE_SPACE_DIMENSIONS == 1
	const double Mass_RBF = 0.333383333333333;
	return (TYPE)powf(fmaxf(0, 1 - inputNormalized), 4) * (4 * inputNormalized + 1) / (Mass_RBF * SuppRBF);

	#elif PHASE_SPACE_DIMENSIONS == 2
	const double Mass_RBF = 0.071428571420238; // this is actually the: int_0^1 phi(r)r dr
	return (TYPE)powf(fmaxf(0, 1 - inputNormalized), 4) * (4 * inputNormalized + 1)
						 / (Mass_RBF * 2 * M_PI * powf(SuppRBF, PHASE_SPACE_DIMENSIONS));

	#elif PHASE_SPACE_DIMENSIONS == 3

	return (TYPE)powf(fmaxf(0, 1 - inputNormalized), 4.00f) * (4.00f * inputNormalized + 1.00f); // We multiply by this last factor to get the L1-normalized RBF

	#else
	return 0;

	#endif

}


#endif