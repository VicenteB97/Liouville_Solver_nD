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

#if DIMENSIONS == 1	//	You have to change this for the 1D normalized RBF
	const FIXED_TYPE Mass_RBF = 0.333383333333333; // this is the: int_0^1 phi(r)dr 
	/// @brief Radial Basis Function interpolation kernel
	/// @param support_radius 
	/// @param x Is the normalized distance, respect to the discretization
	/// @return 
	__device__ inline TYPE RBF(const TYPE support_radius, const TYPE x) {

		return (TYPE)powf(fmaxf(0, 1 - x), 4) * (4 * x + 1) / (Mass_RBF *support_radius); // We multiply by this last factor to get the L1-normalized RBF

	}
#elif DIMENSIONS == 2
	const FIXED_TYPE Mass_RBF = 0.071428571420238; // this is actually the: int_0^1 phi(r)r dr
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

	const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

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


// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //
// ================================================================================================= //

// __global__ void BIN_PARTICLES(	gridPoint* Particles, 
// 								uint32_t* Bins, 
// 								uint32_t* Bin_counter, 
// 								const gridPoint* lowest_node, 
// 								const double discretization_length, 
// 								const uint32_t PtsPerDimension,
// 								const uint32_t Particle_count){

// 	uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

// 	if(globalID < Particle_count){

// 		// Find nearest grid node
// 		uint32_t nearest_idx = 0;
// 		#pragma unroll
// 		for (uint16_t d = 0; d < DIMENSIONS; d++){
// 			nearest_idx += roundf((T) (Particles.dim[d] - lowest_node.dim[d]) / discretization_length) * pow(PtsPerDimension, d);
// 		}	

// 		uint16_t counter_prev = atomicAdd(&Bin_counter[nearest_idx], 1);
// 		Bins[nearest_idx + counter]
// 	}

// }

// template<class T>
// /// @brief Exhaustive PP search (in the respective parameter sample neighborhood)
// /// @param Search_Particles 
// /// @param Fixed_Particles 
// /// @param Index_Array 
// /// @param Matrix_Entries 
// /// @param Num_Neighbors 
// /// @param max_neighbor_num 
// /// @param Adapt_Points 
// /// @param Total_Particles 
// /// @param search_radius 
// /// @return 
// __global__ void __BIN_Exh_PP_Search(const gridPoint* Search_Particles, 
// 									const gridPoint* Fixed_Particles, 
// 									int32_t* Index_Array, 
// 									T* Matrix_Entries, 
// 									int32_t* Num_Neighbors, 
// 									const int32_t max_neighbor_num, 
// 									const int32_t Adapt_Points, 
// 									const int32_t Total_Particles, 
// 									const T search_radius,
// 									const gridPoint* Boundary) {

// 	const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

// 	if (i < Total_Particles) {		
// 		const gridPoint FP_aux	= Fixed_Particles[i];
// 		int32_t			aux		= 1;
// 		const uint32_t	i_aux	= floorf(i / Adapt_Points);										// Tells me what parameter sample I'm at
// 		const int32_t	k		= i * max_neighbor_num;
// 		T				dist;

// 		Index_Array[k]		= i;
// 		Matrix_Entries[k]	= RBF(search_radius, 0);

// 		if(__is_in_domain(FP_aux, Boundary)){
// 			for (uint32_t j = i_aux * Adapt_Points; j < (i_aux + 1) * Adapt_Points; j++) {		// neighborhood where I'm searching

// 				dist = Distance(Search_Particles[j], FP_aux) / search_radius;	// normalized distance between particles

// 				if (dist <= 1 && aux < max_neighbor_num && j != i && __is_in_domain(Search_Particles[j], Boundary)) {
// 					Index_Array[k + aux] = j;
// 					Matrix_Entries[k + aux] = RBF(search_radius, dist);
// 					aux++;
// 				}
// 			}
// 			Num_Neighbors[i] = aux;
// 		}
// 	}
// }

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
		const uint32_t n  = Interaction_Lengths[i];	// total neighbors to look at
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
											const uint16_t					max_steps,
											const T							in_tolerance) {


	// Determine threads and blocks for the simulation
	const uint16_t Threads = (uint32_t)fminf(THREADS_P_BLK, Total_Particles);
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

	if(__is_in_domain(particle, Boundary)){
		
		T dist;

	// I want to compute the index of the lowest neighboring grid node (imagine the lowest corner of a box) and build its nearest neighbors
		int32_t 	lowest_idx = 0;
		gridPoint	temp_gridNode;
		gridPoint	fixed_gridNode = lowest_node;
		
		#pragma unroll
		for (uint16_t d = 0; d < DIMENSIONS; d++){
			int32_t temp_idx = roundf((T) (particle.dim[d] - lowest_node.dim[d]) / grid_discretization_length) - floorf((T) search_radius / grid_discretization_length);
			lowest_idx += temp_idx * pow(PtsPerDimension, d);

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

const u_int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

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
		for (u_int16_t d = 0; d < DIMENSIONS; d++){
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
		for(u_int32_t j = 1; j < num_neighbors_per_particle; j++){

			int32_t idx 	= lowest_idx;
			temp_gridNode 	= fixed_gridNode;

			#pragma unroll
			for (u_int16_t d = 0; d < DIMENSIONS; d++){
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

#endif