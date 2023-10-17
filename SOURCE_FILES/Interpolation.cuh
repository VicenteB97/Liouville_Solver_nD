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

#include "Classes.cuh"

__device__ inline TYPE RBF(const TYPE support_radius, const TYPE x);

/// @brief (GLOBAL FUNCTION) Computes the projection of the particles that have been advected
/// @tparam DIM
/// @tparam T
/// @param particles Array that stores the components of the particles
/// @param projections Array that stores the projections of the previous components in "dimension" component
/// @param totalParticles Total number of particles
/// @param dimension Component where I want to find the dimension
/// @return 
template<uint16_t DIM, class T> __global__ void findProjection(const gridPoint<DIM, T>* particles, T* projections, const UINT totalParticles, const UINT dimension) {
	
	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= totalParticles) { return; }

	projections[globalID] = particles[globalID].dim[dimension];
}

/// @brief 
/// @tparam DIM
/// @tparam T
/// @param Bin_locations 
/// @param Bin_count 
/// @param Bin_local_accSum 
/// @param Search_Particles 
/// @param Bounding_Box 
/// @param Total_Particles 
/// @return 
template<uint16_t DIM, class T> __global__ void Bin_Insertion_Count(UINT*		Bin_locations,
																	UINT*		Bin_count,
																	UINT*		Bin_local_accSum,
																	const gridPoint<DIM, T>* Search_Particles,
																	const grid<DIM, T>	Bounding_Box,
																	const UINT	Total_Particles){
		
	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= Total_Particles) { return; }

	gridPoint<DIM, T> temp_GP = Search_Particles[globalID];

	// Find the Bin where it belongs. We use an unsigned int because since we are working with the bounding box,
	// we're never going to have a point outside. Therefore, there will be a positive index for the bin location.

	UINT bin_idx = Bounding_Box.Get_binIdx(temp_GP);	// this part should be positive...why aren't we getting a positive number?

	// Now, we've got the nearest bin index. This tells the idx of the bin center containing a given particle
	Bin_locations[globalID] = bin_idx;

	//Add one to the number of particles inside the current bin, and store the previous value!
	Bin_local_accSum[globalID] = atomicAdd(&Bin_count[bin_idx], 1);
}

/// @brief 
/// @tparam DIM
/// @tparam T
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
template<uint16_t DIM, class T> __global__ void Count_sort( const gridPoint<DIM, T>*fixed_Particles,
															gridPoint<DIM, T>*		Particles_2sort, 
															const T*				fixed_values, 
															T*						values_2sort,
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

/// @brief 
/// @tparam DIM
/// @tparam T
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
template<uint16_t DIM, class T> __global__ void Neighbor_search(gridPoint<DIM, T>*		Search_Particles,
																const UINT* Bin_count,
																INT*		Index_Array,
																T*				Matrix_Entries,
																UINT*		Num_Neighbors,
																const UINT	max_neighbor_num,
																const UINT	Total_Particles,
																const UINT	offset,
																const T			search_distance,			// This tells us how many discretizations we have to move back to find initial bin to search from
																const grid<DIM, T>		Bounding_Box) {

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// MAYBE: We could use shared memory to access particles nearby (same/near bins)!
	if (globalID >= Total_Particles) { return; }

// Go through the corresponding bins and their particles! Note that we've got to visit some nodes in every direction
// That includes left and right (1D) + up and down (if 2D) + front and behind (if 3D) + ...
// But! Since we know the particles assigned in each bin and how many particles there are in each bin, we can visit efficiently!
	
// Read the input from the sorted Particle array
	gridPoint<DIM, T> fixed_GP = Search_Particles[globalID + offset];

	UINT temp_counter = 0;
	UINT temp_ID = (globalID + offset) * max_neighbor_num;

// Now, we go to the lowest bin and we search all the way up!
	const UINT Bin_offset = lroundf(DISC_RADIUS);

// We find the lowest corner in the neighboring bins (Hence, the Bin_offset variable)
	UINT bin_idx = Bounding_Box.Get_binIdx(fixed_GP, -Bin_offset);

	for (UINT k = 0; k < pow(2 * Bin_offset + 1, DIMENSIONS); k++) { // That's the total number of bins to visit

		// First: find the global index of the bin to search!
		INT aux_index = bin_idx;
		for (uint16_t d = 0; d < DIMENSIONS; d++) {
			aux_index += floorf(positive_rem(k, (UINT)pow(2 * Bin_offset + 1, d + 1)) / (UINT)pow(2 * Bin_offset + 1, d)) * (UINT)pow(Bounding_Box.Nodes_per_Dim, d);
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

				gridPoint<DIM, T>	temp_GP = Search_Particles[m + offset];
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
	Num_Neighbors[globalID + offset] = temp_counter;
}

// Counting sort! This operation orders the grid<DIM, T> indeces and it sets the number of nodes inside each bin

/// @brief 
/// @tparam T 
/// @tparam DIM 
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
template<uint16_t DIM, class T> __host__ int16_t _CS_Neighbor_Search(thrust::device_vector<gridPoint<DIM, T>>& Search_Particles,
																	thrust::device_vector<T> &PDF_vals,
																	thrust::device_vector<INT>& Index_Array,
																	thrust::device_vector<T>& Matrix_Entries,
																	thrust::device_vector<UINT>& Num_Neighbors,
																	const UINT Adapt_Points, 
																	const UINT max_neighbor_num,
																	const grid<DIM, T>& Bounding_Box,
																	const T search_radius) {

	// We need our particles
	UINT Total_Particles = Search_Particles.size();

	thrust::device_vector<UINT> Bin_locations(Adapt_Points,0);

	thrust::device_vector<gridPoint<DIM, T>> temp_Particles(Adapt_Points);		// this one will store the ordered particles!
	thrust::device_vector<T>				 temp_values(Adapt_Points,0);		// this one will store the values of the particles in the new order!

	// Create a bounding box covering the same area as the bounding box from the particles, but change the number of nodes per dim!
	grid<DIM, T> CS_BBox(Bounding_Box);
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
		Bin_Insertion_Count<DIM, T> << <Threads, Blocks >> > (rpc(Bin_locations, 0), 
															rpc(Bin_globalCount, 0),
															rpc(Bin_localCount, 0),
															rpc(Search_Particles, offset), 
															CS_BBox, 
															Adapt_Points);
		gpuError_Check(cudaDeviceSynchronize());

	// Prefix sum of the bin-count array, which gives the cumulative points in each bin (without including the bin itself; thus the name exclusive)
		thrust::exclusive_scan(thrust::device, Bin_globalCount.begin(), Bin_globalCount.end(), Bin_globalCount.begin());

	// Counting sort...we want to sort Search_Particles!
		Count_sort<DIM,T> << <Threads, Blocks >> > (rpc(Search_Particles, 0), 
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
		Neighbor_search<DIM,T> << <Threads, Blocks >> >(rpc(Search_Particles, 0),
													rpc(Bin_globalCount, 0),
													rpc(Index_Array, 0),
													rpc(Matrix_Entries, 0),
													rpc(Num_Neighbors, 0),
													max_neighbor_num,
													Adapt_Points,
													offset,
													search_radius,
													CS_BBox);
		gpuError_Check(cudaDeviceSynchronize());
	}
	return 0;
}

/// @brief 
/// @tparam DIM
/// @tparam T
/// @param Search_Particles 
/// @param Fixed_Particles 
/// @param Index_Array 
/// @param Matrix_Entries 
/// @param Num_Neighbors 
/// @param max_neighbor_num 
/// @param Adapt_Points 
/// @param Total_Particles 
/// @param search_radius 
/// @return 
template<uint16_t DIM, class T> __global__ void Exh_PP_Search(const gridPoint<DIM, T>* Search_Particles,
															const gridPoint<DIM, T>* Fixed_Particles,
															INT* Index_Array,
															T* Matrix_Entries,
															UINT* Num_Neighbors,
															const INT max_neighbor_num,
															const UINT Adapt_Points,
															const UINT Total_Particles,
															const T search_radius) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Total_Particles) { return; }

	gridPoint<DIM, T>		FP_aux	= Fixed_Particles[i];									// Tells me what parameter sample I'm at
	const INT	k			= i * max_neighbor_num;

	Index_Array[k]	= i;
	Matrix_Entries[k] = RBF(search_radius, 0);

	UINT		aux = 1;
	const INT	i_aux = (double)floor((double)i / Adapt_Points);	// this double is used so that the compiler knows that we are using the CUDA version
	T			dist;

	for (UINT j = i_aux * Adapt_Points; j < (i_aux + 1) * Adapt_Points; j++) {		// neighborhood where I'm searching

		gridPoint<DIM, T> Temp_Particle = Search_Particles[j];

		dist = Temp_Particle.Distance(FP_aux) / search_radius;	// normalized distance between particles

		if (dist <= 1 && aux < max_neighbor_num && j != i) {
			Index_Array[k + aux] = j;
			Matrix_Entries[k + aux] = RBF(search_radius, dist);
			aux++;
		}
	}
	Num_Neighbors[i] = aux;
}

//---------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------//


template<class T>
__global__ void UPDATE_VEC(T* x, const T* x0, const T scalar, const T* v, const INT Max_Length) {
	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Max_Length) {
		x[i] = x0[i] + scalar * v[i];
	}
}



template<class T>
__global__ void MATRIX_VECTOR_MULTIPLICATION(T* X, const T* x0, const INT* Matrix_idxs, const T* Matrix_entries, const INT total_length, const UINT* Interaction_Lengths, const INT Max_Neighbors) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;	// For each i, which represents the matrix row, we read the index positions and multiply against the particle weights

	if (i >= total_length) { return; }						// total length = adapt_points * total_samples

// 1.- Compute A*X0										
	// 1.1.- Determine where my particles are!!
	const UINT n  = Interaction_Lengths[i];	// total neighbors to look at
	const UINT i0 = i * Max_Neighbors;		// where does my search index start

	T a = 0;					// auxiliary value for sum (the diagonal is always 1 in our case)

	// 1.2.- Multiply row vec (from matrix) - column vec (possible solution)
	for (UINT j = i0; j < i0 + n; j++) {
		INT p = Matrix_idxs[j];

		a += Matrix_entries[j] * x0[p]; 		// < n calls to global memory
	}

// 2.- Output
	X[i] = a;								// particle weights
}


template<class T>
__host__ int16_t CONJUGATE_GRADIENT_SOLVE(	thrust::device_vector<T>&		GPU_lambdas,
									thrust::device_vector<INT>& GPU_Index_array,
									thrust::device_vector<T>&		GPU_Mat_entries,
									thrust::device_vector<UINT>& GPU_Num_Neighbors,
									thrust::device_vector<T>&		GPU_AdaptPDF,
									const INT					Total_Particles,
									const INT					MaxNeighborNum,
									const UINT					max_steps,
									const T							in_tolerance) {


	// Determine threads and blocks for the simulation
	const UINT Threads = (UINT)fminf(THREADS_P_BLK, Total_Particles);
	const UINT Blocks  = (UINT)floorf((Total_Particles - 1) / Threads) + 1;

	int16_t output = 0;

	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------------------- //
	thrust::device_vector<T>	GPU_R	(Total_Particles);		// residual vector
	thrust::device_vector<T>	GPU_temp(Total_Particles);		// auxiliary vector for computation storage
	thrust::device_vector<T>	GPU_AP	(Total_Particles);		// vector for storing the A*P multiplication
	thrust::device_vector<T>	GPU_P	(Total_Particles);		// P vector
	// ------------------------------------------------------------------------------------------ //

	// Auxiliary values
	INT  k	= 1;	// to control possible overflow of iterations
	bool flag	= true;	// to stop the iterations

// Initialize Conjugate gradient method ----------------------------------------------------
	// Compute A * X0
	MATRIX_VECTOR_MULTIPLICATION <T> << < Blocks, Threads >> > (rpc(GPU_temp, 0), rpc(GPU_lambdas,0), rpc(GPU_Index_array,0),
		rpc(GPU_Mat_entries,0), Total_Particles, rpc(GPU_Num_Neighbors,0), MaxNeighborNum);
	gpuError_Check(cudaDeviceSynchronize());

	// Compute R=B-A*X0
	UPDATE_VEC <T> << <Blocks, Threads >> > (rpc(GPU_R,0), rpc(GPU_AdaptPDF,0), (T)-1, rpc(GPU_temp,0), Total_Particles);
	gpuError_Check(cudaDeviceSynchronize());

	T Alpha, R0_norm, r_norm, aux, beta;

	GPU_P = GPU_R;

	while (flag) { // this flag is useful to know when we have arrived to the desired tolerance
	// Alpha computation (EVERYTHING IS CORRECT!)
		// 1.1.- Compute AP=A*P
		MATRIX_VECTOR_MULTIPLICATION <T> << < Blocks, Threads >> > (rpc(GPU_AP,0), rpc(GPU_P,0), rpc(GPU_Index_array,0),
			rpc(GPU_Mat_entries,0), Total_Particles, rpc(GPU_Num_Neighbors,0), MaxNeighborNum);
		gpuError_Check(cudaDeviceSynchronize());

		// 1.2.- Compute P'*AP
		thrust::transform(GPU_P.begin(), GPU_P.end(), GPU_AP.begin(), GPU_temp.begin(), thrust::multiplies<T>());
		aux = thrust::reduce(thrust::device, GPU_temp.begin(), GPU_temp.end());

		// 1.3.- R'*R
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_temp.begin(), thrust::multiplies<T>());
		R0_norm = thrust::reduce(thrust::device, GPU_temp.begin(), GPU_temp.end());

		Alpha = R0_norm / aux;

		// New X and R: (new, old, scalar, driving vec, total length)
		// 1.- Update Lambdas
		UPDATE_VEC <T> << <Blocks, Threads >> > (rpc(GPU_lambdas,0), rpc(GPU_lambdas,0), Alpha, rpc(GPU_P,0), Total_Particles);
		// we DO NOT use cudaDeviceSynchronize() because the following CUDA kernel does not require this kernel to be done...we may save a (very small) amount of time

		// 2.- Update residuals 
		UPDATE_VEC <T> << <Blocks, Threads >> > (rpc(GPU_R,0), rpc(GPU_R,0), -Alpha, rpc(GPU_AP,0), Total_Particles);
		gpuError_Check(cudaDeviceSynchronize());

		// Compute residual l_2 norm
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_temp.begin(), thrust::multiplies<T>());
		r_norm = thrust::reduce(thrust::device, GPU_temp.begin(), GPU_temp.end()); // sum of its elements
		r_norm = sqrt(r_norm);

		if ((T) r_norm / Total_Particles < in_tolerance) {
			#if OUTPUT_INFO == 1
				std::cout << "Convergence. Iteration count: " << k << "\n";
			#endif

			flag = false;
			output = 0;
			break;
		}
		else if (k > max_steps) {
			std::cout << "No convergence was obtained after reaching max. allowed iterations. Last residual norm was: " << r_norm << "\n";
			std::cout << "+-------------------------------------------------------------------+\n";

			std::cin.get();
			output = -1;
			flag = false;
			break;
		}
		else {
			beta = r_norm * r_norm / R0_norm;

			UPDATE_VEC <T> << <Blocks, Threads >> > (rpc(GPU_P,0), rpc(GPU_R,0), beta, rpc(GPU_P,0), Total_Particles);
			gpuError_Check(cudaDeviceSynchronize());
			k++;
		}
	}
	return output;
}



template<uint16_t DIM, class T>
__global__ void RESTART_GRID_FIND_GN(gridPoint<DIM, T>* Particle_Positions,
									T* PDF,
									T* lambdas,
									const Param_pair* Parameter_Mesh,
									const INT* n_Samples,
									const T 	 		search_radius,
									const UINT	 		Adapt_Pts,
									const UINT	 		Block_samples,
									const UINT	 		offset,
									const grid<DIM, T> 	Mesh,
									const grid<DIM, T>	Underlying_Mesh) {
	
	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Adapt_Pts * Block_samples) { return; }

	UINT		Current_sample = offset + floorf(i / Adapt_Pts);
	Param_vec	aux = Gather_Param_Vec(Current_sample, Parameter_Mesh, n_Samples);

	T weighted_lambda = lambdas[i] * aux.Joint_PDF;

	gridPoint<DIM, T>	particle = Particle_Positions[i];

	// Find the point in the lowest corner of the search box!
	gridPoint<DIM, T> Lowest_node = Underlying_Mesh.Get_node(Underlying_Mesh.Get_binIdx(particle, -lround(DISC_RADIUS)));

	const INT Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;

	// Go through all the nodes where rewriting will be possible
	for (uint16_t k = 0; k < pow(Neighbors_per_dim, DIM); k++) {

		gridPoint<DIM, T> visit_node = Lowest_node;

		// Get the node at that point
		for (uint16_t d = 0; d < DIM; d++) {
			INT temp_idx = floor(positive_rem(k, pow(Neighbors_per_dim, d + 1)) / pow(Neighbors_per_dim, d));
			visit_node.dim[d] += temp_idx * Mesh.Discr_length();
		}

		// If it is inside our problem mesh...
		if (Mesh.Contains_particle(visit_node)) {

			// Calculate normalized distance
			T dist = visit_node.Distance(particle) / search_radius;

			// if it's inside the RBF support...
			if (dist <= 1) {

				dist = RBF(search_radius, dist) * weighted_lambda;

				INT idx = Mesh.Get_binIdx(visit_node);

				atomicAdd(&PDF[idx], dist);
			}

		}
	}
}


template<uint16_t DIM, class T>
__global__ void RESTART_GRID_FIND_GN(gridPoint<DIM, T>*	Particle_Positions,
							float*				PDF,
							float*				lambdas,
							const Impulse_Param_vec* Impulse_weights,
							const T 			search_radius,
							const UINT	 		Adapt_Pts,
							const UINT	 		Current_sample,
							const grid<DIM, T>	Mesh,
							const grid<DIM, T>	Underlying_Mesh) {
	// OUTPUT: New values of the PDF at the fixed grid<DIM, T>

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Adapt_Pts) { return; }

	T weighted_lambda = lambdas[i + Current_sample * Adapt_Pts] * Impulse_weights[Current_sample].Joint_PDF;				// the specific sample weight

	gridPoint<DIM, T>	particle(Particle_Positions[i + Current_sample * Adapt_Pts]);

	// Find the point in the lowest corner of the search box!
	gridPoint<DIM, T> Lowest_node = Underlying_Mesh.Get_node(Underlying_Mesh.Get_binIdx(particle, -roundf(DISC_RADIUS)));

	const INT Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;

	// Go through all the nodes where rewriting will be possible
	for (uint16_t k = 0; k < pow(Neighbors_per_dim, DIM); k++) {

		gridPoint<DIM, T> visit_node = Lowest_node;

		// Get the node at that point
		for (uint16_t d = 0; d < DIM; d++) {
			INT temp_idx = floor(positive_rem(k, pow(Neighbors_per_dim, d + 1)) / pow(Neighbors_per_dim, d));
			visit_node.dim[d] += temp_idx * Mesh.Discr_length();
		}

		// If it is inside our problem mesh...
		if (Mesh.Contains_particle(visit_node)) {

			// Calculate normalized distance
			T dist = visit_node.Distance(particle) / search_radius;

			// if it's inside the RBF support...
			if (dist <= 1) {

				dist = RBF(search_radius, dist) * weighted_lambda;

				INT idx = Mesh.Get_binIdx(visit_node);

				atomicAdd(&PDF[idx], dist);
			}

		}
	}
}


/// @brief This function makes sure that the PDF does not have any negative values! (Having this enforced by the interpolation would've been wonderful)
/// @param PDF 
/// @param Grid_Nodes 
/// @return 
template<class T>
__global__ void CORRECTION(T* PDF, const INT Grid_Nodes){
	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Grid_Nodes){
		PDF[i] = fmaxf(PDF[i], 0.00f);
	}
}


// L1-normalized RBF Function definitions

#if DIMENSIONS == 1	//	You have to change this for the 1D normalized RBF
	#define Mass_RBF 0.333383333333333 // this is the: int_0^1 phi(r)dr 
	/// @brief Radial Basis Function interpolation kernel
	/// @param support_radius 
	/// @param x Is the normalized distance, respect to the discretization
	/// @return 
	__device__ inline TYPE RBF(const TYPE support_radius, const TYPE x) {
		return (TYPE)powf(fmaxf(0, 1 - x), 4) * (4 * x + 1) / (Mass_RBF *support_radius); // We multiply by this last factor to get the L1-normalized RBF
	}

#elif DIMENSIONS == 2
	#define Mass_RBF 0.071428571420238 // this is actually the: int_0^1 phi(r)r dr
	/// @brief Radial Basis Function interpolation kernel
	/// @param support_radius 
	/// @param x Is the normalized distance, respect to the discretization
	/// @return 
	__device__ inline TYPE RBF(const TYPE support_radius, const TYPE x) {
		return (TYPE)powf(fmaxf(0, 1 - x), 4.00f) * (4.00f * x + 1.00f) / (Mass_RBF * 2.00f * M_PI * powf(support_radius, (TYPE)DIMENSIONS)); // We multiply by this last factor to get the L1-normalized RBF
	}
	
#elif DIMENSIONS == 3	// ADD THE TOTAL MASS FOR EACH 3D RBF FUNCTION
	__device__ inline TYPE RBF(const TYPE support_radius, const TYPE x) {
		return (TYPE)powf(fmaxf(0, 1 - x), 4.00f) * (4.00f * x + 1.00f); // We multiply by this last factor to get the L1-normalized RBF
	}
#else
	std::cout << "Error in 'Interpolation.cuh'. You are choosing an unavailable option. Go back to 'Case_definition.cuh' and re-check options for DIMENSIONS.\n"
	return -1;
#endif


#endif