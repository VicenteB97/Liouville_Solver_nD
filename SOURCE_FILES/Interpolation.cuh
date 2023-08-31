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

template<class T>
__global__ 
void Bin_Insertion_Count(uint32_t*	Bin_locations,
						uint32_t*		Bin_count,
						uint32_t*		Bin_local_accSum,
						const gridPoint* Search_Particles,
						const T			grid_discretization,
						const uint32_t	Total_Particles,
						const uint32_t	PtsPerDimension,
						const gridPoint lowest_node){
		
	const uint64_t globalId = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalId < Total_Particles) {
		gridPoint temp = Search_Particles[globalId];

		// Find the Bin where it belongs
		uint32_t bin_idx = temp.GetBin(grid_discretization, 0, lowest_node, PtsPerDimension);

		// Now, we've got the nearest bin index. This tells the idx of the bin center containing a given particle
		Bin_locations[globalId] = bin_idx;

		//Add one to the number of particles inside the current bin, and store the previous value!
		Bin_local_accSum[globalId] = atomicAdd(&Bin_count[bin_idx], 1);
	}
}

template<class T>
__global__ 
void Count_sort(const gridPoint* fixed_Particles, 
				gridPoint* Particles_2sort, 
				const T* fixed_values, 
				T *values_2sort,
				const uint32_t* Bin_count, 
				const uint32_t* Bin_locations,
				const uint32_t* Bin_local_accSum,	
				const uint32_t Total_Particles) {

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID < Total_Particles) {
		// read bin location
		uint32_t my_bin_idx = Bin_locations[globalID];

		// read local cumulative sum
		uint32_t my_count_loc = Bin_local_accSum[globalID];

		// read the prefix sum of the bin!
		uint32_t pref_sum_idx = Bin_count[my_bin_idx];

		// Add to ordered particle array (pref. sum tells the first indx of the particles to be added, where local_count)
		// tells the position among those particles
		uint32_t final_idx = pref_sum_idx + my_count_loc;
		Particles_2sort[final_idx]	= fixed_Particles[globalID];
		values_2sort[final_idx]		= fixed_values[globalID];
	}
}

template<class T>
__global__ void Neighbor_search(gridPoint*		Search_Particles,
								const uint32_t* Bin_count,
								int32_t*		Index_Array,
								T*				Matrix_Entries,
								uint32_t*		Num_Neighbors,
								const uint32_t	max_neighbor_num,
								const gridPoint lowest_node, 
								const uint32_t	Total_Particles,
								const uint32_t	PtsPerDimension,
								const T			search_distance,			// This tells us how many discretizations we have to move back to find initial bin to search from
								const T			grid_discretization,
								const gridPoint* Boundary) {

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// MAYBE: We could use shared memory to access particles nearby (same/near bins)!

	if (globalID < Total_Particles) {

// Go through the corresponding bins and their particles! Note that we've got to visit some nodes in every direction
// That includes left and right (1D) + up and down (if 2D) + front and behind (if 3D) + ...
// But! Since we know the particles assigned in each bin and how many particles there are in each bin, we can visit efficiently!

		// Read the input from the sorted Particle array
		gridPoint fixed_GP = Search_Particles[globalID];

		uint64_t temp_counter = 0;
		uint64_t temp_ID = globalID * max_neighbor_num;

		// If the particle is not in the domain, we ignore it
		if (!fixed_GP.is_in_domain(Boundary)) {
			Num_Neighbors[globalID] = 1;
			Matrix_Entries[temp_ID] = RBF(search_distance, 0);
			Index_Array[temp_ID]	= globalID;
			return;
		}

		// Now, we go to the lowest bin and we search all the way up!
		const uint32_t Bin_offset = floorf(DISC_RADIUS);

		// We find the lowest corner in the neighboring bins (Hence, the Bin_offset variable)
		uint32_t bin_idx = fixed_GP.GetBin(grid_discretization, -Bin_offset, lowest_node, PtsPerDimension);

		for (uint32_t k = 0; k < powf(2 * Bin_offset + 1, DIMENSIONS); k++) { // That's the total number of bins to visit

		// First: find the global index of the bin to search!
			uint64_t aux_index = bin_idx;

			#pragma unroll
			for (uint32_t d = 0; d < DIMENSIONS; d++) {
				aux_index += floorf(positive_rem(k, powf(2 * Bin_offset + 1, d + 1)) / powf(2 * Bin_offset + 1, d)) * powf(PtsPerDimension, d);
			}

		// Now that we have the bin to search, all we have to do is go through the particles that are in the corresponding bin:
			uint32_t startIdx	= Bin_count[aux_index];
			uint32_t endIdx		= Bin_count[aux_index + 1];

		// Add the points in the current bin
			for (int32_t m = startIdx; m < endIdx; m++) {

				gridPoint	temp_GP = Search_Particles[m];
				TYPE		dist = Distance(fixed_GP, temp_GP) / search_distance;		// HERE PROBLEM

				if (dist <= 1 && temp_GP.is_in_domain(Boundary) && temp_counter < max_neighbor_num) {
					// include particle index into COO-sparse information
					// include RBF-distance value into matrix values
					// add count to number of neighbors (maybe optional)
					Index_Array[temp_ID + temp_counter]		= m;
					Matrix_Entries[temp_ID + temp_counter]	= RBF(search_distance, dist);
					temp_counter++;
				}
			}
		}
		Num_Neighbors[globalID] = temp_counter;
	}
}

// Counting sort! This operation orders the grid indeces and it sets the number of nodes inside each bin
template<class T>
int16_t _CS_Neighbor_Search(thrust::device_vector<gridPoint>& Search_Particles,
							thrust::device_vector<T> &PDF_vals,
							thrust::device_vector<int32_t>& Index_Array,
							thrust::device_vector<T>& Matrix_Entries,
							thrust::device_vector<uint32_t>& Num_Neighbors,
							const uint32_t PtsPerDimension, 
							const uint32_t Adapt_Points, 
							const gridPoint lowest_node,
							const uint32_t max_neighbor_num,
							const T search_radius,
							const T grid_discretization,
							const thrust::device_vector<gridPoint>& Boundary) {

	// We need our particles
	uint32_t Total_Particles = Search_Particles.size();
	uint32_t Grid_Nodes		 = pow(PtsPerDimension, DIMENSIONS);

	thrust::device_vector<uint32_t> Bin_locations(Adapt_Points,0);
	thrust::device_vector<uint32_t> Bin_localSum(Adapt_Points,0);
	thrust::device_vector<uint32_t> Bin_count(Grid_Nodes, 0);

	thrust::device_vector<gridPoint> temp_Particles(Adapt_Points);		// this one will store the ordered particles!
	thrust::device_vector<T>		 temp_values(Adapt_Points,0);		// this one will store the values of the particles in the new order!

	// Bin the particles!
	uint32_t Threads = fmin(THREADS_P_BLK, Adapt_Points);
	uint32_t Blocks = floor((Adapt_Points - 1) / Threads) + 1;

	for (uint32_t k = 0; k < Total_Particles / Adapt_Points; k++) { // We've got to repeat this once per random sample...we'll see another way out...

		uint32_t offset = k * Adapt_Points;

		// maybe you have to load the lowest node as a thrust vector
		Bin_Insertion_Count<T> << <Threads, Blocks >> > (rpc(Bin_locations, 0), 
														rpc(Bin_count, 0), 
														rpc(Bin_localSum, 0), 
														rpc(Search_Particles, offset), 
														grid_discretization, 
														Adapt_Points, 
														PtsPerDimension, 
														lowest_node);
		gpuError_Check(cudaDeviceSynchronize());

		// Prefix sum of the bin-count array, which gives the cumulative points in each bin
		thrust::exclusive_scan(thrust::device, Bin_count.begin(), Bin_count.end(), Bin_count.begin());

		// Counting sort...we want to sort Search_Particles!
		Count_sort<T> << <Threads, Blocks >> > (rpc(Search_Particles, offset), 
												rpc(temp_Particles, 0), 
												rpc(PDF_vals,offset),
												rpc(temp_values,0),
												rpc(Bin_count, 0), 
												rpc(Bin_locations, 0), 
												rpc(Bin_localSum, 0), 
												Adapt_Points);
		gpuError_Check(cudaDeviceSynchronize());

		thrust::copy(temp_Particles.begin(), temp_Particles.end(), &Search_Particles[offset]);
		thrust::copy(temp_values.begin(), temp_values.end(), &PDF_vals[offset]);

		Neighbor_search<T> << <Threads, Blocks >> >(rpc(Search_Particles, offset),
													rpc(Bin_count, 0),
													rpc(Index_Array, offset),
													rpc(Matrix_Entries, offset),
													rpc(Num_Neighbors, offset),
													max_neighbor_num,
													lowest_node,
													Adapt_Points,
													PtsPerDimension,
													search_radius,
													grid_discretization,
													rpc(Boundary, 0));
		gpuError_Check(cudaDeviceSynchronize());
	}
	return 0;
}

template<class T>
/// @brief Exhaustive PP search (in the respective parameter sample neighborhood)
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
__global__ void Exh_PP_Search(	const gridPoint* Search_Particles, 
								const gridPoint* Fixed_Particles, 
								int32_t* Index_Array, 
								T* Matrix_Entries, 
								uint32_t* Num_Neighbors, 
								const int32_t max_neighbor_num, 
								const int32_t Adapt_Points, 
								const int32_t Total_Particles, 
								const T search_radius,
								const gridPoint* Boundary) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles) {		
		const gridPoint FP_aux	= Fixed_Particles[i];
		uint32_t			aux	= 1;
		const uint32_t	i_aux	= floorf(i / Adapt_Points);										// Tells me what parameter sample I'm at
		const int64_t	k		= i * max_neighbor_num;
		T				dist;

		Index_Array[k]		= i;
		Matrix_Entries[k]	= RBF(search_radius, 0);

		if(__is_in_domain(FP_aux, Boundary)){
			for (uint64_t j = i_aux * Adapt_Points; j < (i_aux + 1) * Adapt_Points; j++) {		// neighborhood where I'm searching

				gridPoint Temp_Particle = Search_Particles[j];

				dist = Distance(Temp_Particle, FP_aux) / search_radius;	// normalized distance between particles

				if (dist <= 1 && aux < max_neighbor_num && j != i && __is_in_domain(Temp_Particle, Boundary)) {
					Index_Array[k + aux] = j;
					Matrix_Entries[k + aux] = RBF(search_radius, dist);
					aux++;
				}
			}
			Num_Neighbors[i] = aux;
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------//

template<class T>
/// @brief General function for computing the "update" of a vector x0, storing it in x
/// @param x Output vector
/// @param x0 Vec. to be updated
/// @param scalar "Strength" of the updating vector
/// @param v "Direction" of the updating vector
/// @param Max_Length Length of the vectors
/// @return 
__global__ void UPDATE_VEC(T* x, const T* x0, const T scalar, const T* v, const int32_t Max_Length) {
	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Max_Length) {
		x[i] = x0[i] + scalar * v[i];
	}
}

template<class T>
/// @brief This function performs a sparse matrix multiplication
/// @param X 
/// @param x0 
/// @param Matrix_idxs 
/// @param Matrix_entries 
/// @param total_length 
/// @param Interaction_Lengths 
/// @param Max_Neighbors 
/// @return 
__global__ void MATRIX_VECTOR_MULTIPLICATION(T* X, const T* x0, const int32_t* Matrix_idxs, const T* Matrix_entries, const int32_t total_length, const uint32_t* Interaction_Lengths, const int32_t Max_Neighbors) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;	// For each i, which represents the matrix row, we read the index positions and multiply against the particle weights

	if (i < total_length) {						// total length = adapt_points * total_samples
// 1.- Compute A*X0										
	// 1.1.- Determine where my particles are!!
		const uint32_t n = Interaction_Lengths[i];	// total neighbors to look at
		const uint32_t i0 = i * Max_Neighbors;		// where does my search index start

		T 			a = 0;					// auxiliary value for sum (the diagonal is always 1 in our case)
		uint32_t 	p;						// auxiliary variable for indexing
		// 1.2.- Multiply row vec (from matrix) - column vec (possible solution)
		for (uint32_t j = i0; j < i0 + n; j++) {
			p = Matrix_idxs[j];
			a += Matrix_entries[j] * x0[p];		// < n calls to global memory
		}

// 2.- Output
		X[i] = a;								// particle weights
	}
}

template<class T>
/// @brief This function solves a linear system of equations where the matrix is SPARSE, SYMMETRIC AND POSITIVE DEFINITE
/// @param GPU_lambdas 
/// @param GPU_Index_array 
/// @param GPU_Mat_entries 
/// @param GPU_Num_Neighbors 
/// @param GPU_AdaptPDF 
/// @param Total_Particles 
/// @param MaxNeighborNum 
/// @param max_steps 
/// @param in_tolerance 
/// @return An integer for determining whether it has exited successfully or not
__host__ int32_t CONJUGATE_GRADIENT_SOLVE(	thrust::device_vector<T>&		GPU_lambdas,
											thrust::device_vector<int32_t>& GPU_Index_array,
											thrust::device_vector<T>&		GPU_Mat_entries,
											thrust::device_vector<uint32_t>& GPU_Num_Neighbors,
											thrust::device_vector<T>&		GPU_AdaptPDF,
											const int32_t					Total_Particles,
											const int32_t					MaxNeighborNum,
											const int32_t					max_steps,
											const T							in_tolerance) {


	// Determine threads and blocks for the simulation
	const uint32_t Threads = (uint32_t)fminf(THREADS_P_BLK, Total_Particles);
	const uint64_t Blocks  = (uint32_t)floorf((Total_Particles - 1) / Threads) + 1;

	int16_t output = 0;

	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------------------- //
	thrust::device_vector<T>	GPU_R	(Total_Particles);		// residual vector
	thrust::device_vector<T>	GPU_temp(Total_Particles);		// auxiliary vector for computation storage
	thrust::device_vector<T>	GPU_AP	(Total_Particles);		// vector for storing the A*P multiplication
	thrust::device_vector<T>	GPU_P	(Total_Particles);		// P vector
	// ------------------------------------------------------------------------------------------ //

	// Auxiliary values
	int32_t  k	= 1;	// to control possible overflow of iterations
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
			std::cout << "/-------------------------------------------------------------------/\n";

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

template<class T>
/// @brief This function computes the neighboring grid nodes for each advected particle from the simulation.
/// @param Particle_Positions 
/// @param Neighbor_Indx_array 
/// @param search_radius 
/// @param lowest_node 
/// @param grid_discretization_length 
/// @param PtsPerDimension 
/// @param GridNodes 
/// @param Total_Particles 
/// @return 
__global__ void RESTART_GRID_FIND_GN(gridPoint*  	 Particle_Positions,
									T* 		 	PDF,
									T* 		 lambdas,
									const Param_pair* Parameter_Mesh,
									const int32_t* 		 n_Samples,
									const T 	 search_radius,
									const gridPoint  lowest_node,
									const T 	 grid_discretization_length,
									const int32_t	 	 PtsPerDimension,
									const int32_t	 	 Adapt_Pts,
									const int32_t	 	 Block_samples,
									const uint32_t 	 offset,
									const gridPoint* Boundary) {
// OUTPUT: New values of the PDF at the fixed grid

const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < Adapt_Pts * Block_samples) {
	uint32_t num_neighbors_per_dim 	 = 2 * floorf(search_radius / grid_discretization_length) + 1;
	uint32_t num_neighbors_per_particle = pow(num_neighbors_per_dim, DIMENSIONS);
	uint32_t Current_sample			 = offset + floorf(i / Adapt_Pts);

	Param_vec aux = _Gather_Param_Vec(Current_sample, Parameter_Mesh, n_Samples);

	gridPoint 	particle 		= Particle_Positions[i];
			T 	weighted_lambda = lambdas[i] * aux.Joint_PDF;

	if(particle.is_in_domain(Boundary)){
		
		T dist;

	// I want to compute the index of the lowest neighboring grid node (imagine the lowest corner of a box) and build its nearest neighbors
		int32_t 	lowest_idx = particle.GetBin(grid_discretization_length,floorf(DISC_RADIUS),lowest_node,PtsPerDimension);
		gridPoint	temp_gridNode;
		gridPoint	fixed_gridNode = lowest_node;
		
		#pragma unroll
		for (uint32_t d = 0; d < DIMENSIONS; d++){
			int32_t temp_idx = roundf((T) (particle.dim[d] - lowest_node.dim[d]) / grid_discretization_length) - floorf((T) search_radius / grid_discretization_length);
			fixed_gridNode.dim[d] += temp_idx * grid_discretization_length;
		}	
		
		// store the lowest sparse index identification (remember we are alredy storing the transposed matrix. The one we will need for multiplication)
		if (lowest_idx >= 0 && lowest_idx < pow(PtsPerDimension, DIMENSIONS) && __is_in_domain(fixed_gridNode, Boundary)){
			dist = Distance(fixed_gridNode, particle) / search_radius;
			if (dist <= 1){
				dist = RBF(search_radius, dist) * weighted_lambda;
				__atomicAdd(&PDF[lowest_idx], dist);
			}
		}

	// now, go through all the neighboring grid nodes and add the values to the PDF field
		for(uint32_t j = 1; j < num_neighbors_per_particle; j++){
			int32_t idx = lowest_idx;
			temp_gridNode = fixed_gridNode;

			#pragma unroll
			for (uint32_t d = 0; d < DIMENSIONS; d++){
				int32_t temp_idx = floorf( positive_rem(j, pow(num_neighbors_per_dim, d + 1)) / pow(num_neighbors_per_dim, d) ); 
				idx += temp_idx * pow(PtsPerDimension, d);

				temp_gridNode.dim[d] = temp_idx * grid_discretization_length + fixed_gridNode.dim[d];
			}

			if (idx >= 0 && idx < pow(PtsPerDimension, DIMENSIONS) && __is_in_domain(temp_gridNode,Boundary))
			{
				dist = Distance(temp_gridNode, particle) / search_radius;
				if (dist <= 1){
					dist = RBF(search_radius, dist) * weighted_lambda;
					__atomicAdd(&PDF[idx], dist);
				}
			}
			
		}
	}
}
}

__global__ void RESTART_GRID_FIND_GN_II(gridPoint*  Particle_Positions,
										float* 		PDF,
										float* 		lambdas,
										const Impulse_Param_vec* Impulse_weights,
										const float 	search_radius,
										const gridPoint lowest_node,
										const float 	grid_discretization_length,
										const int32_t	 	PtsPerDimension,
										const int32_t	 	Adapt_Pts,
										const int32_t	 	Current_sample,
										const gridPoint* Boundary) {
// OUTPUT: New values of the PDF at the fixed grid

const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < Adapt_Pts) {
	int32_t num_neighbors_per_dim 		= (int32_t) 2 * floorf(search_radius / grid_discretization_length) + 1;
	int32_t num_neighbors_per_particle 	= (int32_t) pow(num_neighbors_per_dim, DIMENSIONS);

	gridPoint 	particle 		= Particle_Positions[i + Current_sample * Adapt_Pts];
	float 		weighted_lambda = lambdas[i + Current_sample * Adapt_Pts] * Impulse_weights[Current_sample].Joint_PDF;				// the specific sample weight

	if(__is_in_domain(particle, Boundary)){
		float dist;

		// I want to compute the index of the lowest neighboring grid node and build its nearest neighbors
		int32_t 	lowest_idx = 0;
		gridPoint	temp_gridNode;
		gridPoint	fixed_gridNode = lowest_node;
		
		#pragma unroll
		for (uint32_t d = 0; d < DIMENSIONS; d++){
			int32_t temp_idx = roundf((float) (particle.dim[d] - lowest_node.dim[d]) / grid_discretization_length) - floorf((float) search_radius / grid_discretization_length);
			lowest_idx += temp_idx * pow(PtsPerDimension, d);

			fixed_gridNode.dim[d] += temp_idx * grid_discretization_length;
		}	
		
		// store the lowest sparse index identification (remember we are alredy storing the transposed matrix. The one we will need for multiplication)
		if (lowest_idx > 0 && lowest_idx < (int32_t) pow(PtsPerDimension, DIMENSIONS) && __is_in_domain(fixed_gridNode, Boundary)){
			dist = Distance(fixed_gridNode, particle) / search_radius;
			if (dist <= 1){
				dist = RBF(search_radius, dist) * weighted_lambda;
				__atomicAdd(&PDF[lowest_idx], dist);
			}
		}

	// now, go through all the neighboring grid nodes and add the values to the PDF field
		for(uint32_t j = 1; j < num_neighbors_per_particle; j++){

			int32_t idx 	= lowest_idx;
			temp_gridNode 	= fixed_gridNode;

			#pragma unroll
			for (uint32_t d = 0; d < DIMENSIONS; d++){
				int32_t temp_idx = (int32_t) floorf( positive_rem(j, (int32_t)pow(num_neighbors_per_dim, d + 1)) / pow(num_neighbors_per_dim, d) ); 
				idx += temp_idx * pow(PtsPerDimension, d);

				temp_gridNode.dim[d] = temp_idx * grid_discretization_length + fixed_gridNode.dim[d];
			}

			if (idx > 0 && idx < (int32_t) pow(PtsPerDimension, DIMENSIONS) && __is_in_domain(temp_gridNode,Boundary))
			{
				dist = Distance(temp_gridNode, particle) / search_radius;
				if (dist <= 1){
					dist = RBF(search_radius, dist) * weighted_lambda;
					__atomicAdd(&PDF[idx], dist);
				}
			}
			
		}
	}
}
}

template<class T>
/// @brief This function makes sure that the PDF does not have any negative values! (Having this enforced by the interpolation would've been wonderful)
/// @param PDF 
/// @param Grid_Nodes 
/// @return 
__global__ void CORRECTION(T* PDF, const int32_t Grid_Nodes){
	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Grid_Nodes){
		PDF[i] = fmaxf(PDF[i], 0.00f);

		if(PDF[i] < 0.000001){PDF[i]=0;}
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
	
#elif DIMENSIONS == 3
	__device__ inline TYPE RBF(const TYPE support_radius, const TYPE x) {
		return (TYPE)powf(fmaxf(0, 1 - x), 4.00f) * (4.00f * x + 1.00f); // We multiply by this last factor to get the L1-normalized RBF
	}
#else
	std::cout << "Error in 'Mat_Ops.cuh'. You are choosing an unavailable option. Go back to 'Case_definition.cuh' and re-check options for DIMENSIONS.\n"
	return -1;
#endif


#endif