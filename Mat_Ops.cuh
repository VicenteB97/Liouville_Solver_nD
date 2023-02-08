#ifndef __MAT_OPS_CUH__
#define __MAT_OPS_CUH__

#include "Classes.cuh"

#define Mass_RBF 0.071428571420238
/// <summary>
/// Compactly Supported Radial Basis Function (CS-RBF): Wendland C2 kernel (L1-normalized)
/// </summary>
/// <param name="entry"> - Euclidean distance / support radius</param>
/// <returns></returns>
///  * 4.5576820034785532
__device__ double RBF(const double support_radius, const double entry) {

	return pow(fmaxf(0, 1 - entry), 4) * (4 * entry + 1) / (Mass_RBF * 2 * M_PI * support_radius * support_radius); // We multiply by this last factor to get the L1-normalized RBF

}

// Define CUDA implementation for the max function
__device__ double cuda_fmax(double X, double Y) {

	if (X >= Y) {
		return X;
	}
	else {
		return Y;
	}

}

/// <summary>
/// This function performs an exhaustive search in the particle set
/// </summary>
/// <param name="Search_Particles"></param>
/// <param name="Fixed_Particles"></param>
/// <param name="Index_Array"></param>
/// <param name="Matrix_Entries"></param>
/// <param name="Num_Neighbors"></param>
/// <param name="max_neighbor_num"></param>
/// <param name="Adapt_Points"></param>
/// <param name="Total_Particles"></param>
/// <param name="search_radius"></param>
/// <returns></returns>
__global__ void Exh_PP_Search(const gridPoint* Search_Particles, const gridPoint* Fixed_Particles, int* Index_Array, double* Matrix_Entries, int* Num_Neighbors, const int max_neighbor_num, const int Adapt_Points, const int Total_Particles, const double search_radius) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles) {
		int				aux		= 1;
		const int		i_aux	= floorf(i / Adapt_Points);										// Tells me what parameter sample I'm at
		const int		k		= i * max_neighbor_num;
		double			dist;
		const gridPoint FP_aux	= Fixed_Particles[i];

		Index_Array[k]		= i;
		Matrix_Entries[k]	= RBF(search_radius, 0);

		for (unsigned int j = i_aux * Adapt_Points; j < (i_aux + 1) * Adapt_Points; j++) {		// neighborhood where I'm searching

			dist = Distance(Search_Particles[j], FP_aux) / search_radius;	// normalized distance between particles

			if (dist <= 1 && aux < max_neighbor_num && j != i) {
				Index_Array[k + aux] = j;
				Matrix_Entries[k + aux] = RBF(search_radius, dist);
				aux++;
			}
		}
		Num_Neighbors[i] = aux;
	}
}

//---------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------------------------------------------------------------------------//

template<class T>
/// <summary>
/// This function allows the computation of the difference of GPU vectors
/// </summary>
/// <param name="output"></param>
/// <param name="u"> - Vector which is to be substracted by v </param>
/// <param name="v"> - Substracting vector</param>
/// <param name="Max_Length"> - Vector lengths </param>
/// <returns></returns>
__global__ void DIFF_VECS(T* output, const T* u, const T* v, const int Max_Length) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Max_Length) {
		output[i] = u[i] - v[i];
	}
}

template<class T>
/// <summary>
/// Update function. This function allows to compute the update of a vector in the CG linear solver
/// </summary>
/// <param name="x"> - Final vector</param>
/// <param name="x0"> - Initial vector (to be updated)</param>
/// <param name="scalar"> - Scalar multiplying the update vector </param>
/// <param name="v"> - Update vector </param>
/// <param name="Max_Length"> - Vector lengths</param>
/// <returns></returns>
__global__ void UPDATE_VEC(T* x, const T* x0, const T scalar, const double* v, const int Max_Length) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

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
__global__ void MATRIX_VECTOR_MULTIPLICATION(T* X, const T* x0, const int* Matrix_idxs, const T* Matrix_entries, const int total_length, const int* Interaction_Lengths, const int Max_Neighbors) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;	// For each i, which represents the matrix row, we read the index positions and multiply against the particle weights

	if (i < total_length) {						// total length = adapt_points * total_samples
// 1.- Compute A*X0										
	// 1.1.- Determine where my particles are!!
		const int n = Interaction_Lengths[i];	// total neighbors to look at
		const int i0 = i * Max_Neighbors;		// where does my search index start

		T a = 0;							// auxiliary value for sum (the diagonal is always 1 in our case)
		int p;									// auxiliary variable for indexing
	// 1.2.- Multiply row vec (from matrix) - column vec (possible solution)
		for (unsigned int j = i0; j < i0 + n; j++) {
			p = Matrix_idxs[j];
			a += Matrix_entries[j] * x0[p];
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
__host__ int CONJUGATE_GRADIENT_SOLVE(	thrust::device_vector<T>&	GPU_lambdas,
										thrust::device_vector<int>& GPU_Index_array,
										thrust::device_vector<T>&	GPU_Mat_entries,
										thrust::device_vector<int>& GPU_Num_Neighbors,
										thrust::device_vector<T>&	GPU_AdaptPDF,
										const int					Total_Particles,
										const int					MaxNeighborNum,
										const int					max_steps,
										const T						in_tolerance) {


	// Determine threads and blocks for the simulation
	const int Threads = (int)fminf(THREADS_P_BLK, Total_Particles);
	const int Blocks  = (int)floorf((Total_Particles - 1) / Threads) + 1;

	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------------------- //
	thrust::device_vector<T>	GPU_R	(Total_Particles);		// residual vector
	thrust::device_vector<T>	GPU_AUX (Total_Particles);		// auxiliary vector for computation storage
	thrust::device_vector<T>	GPU_AP	(Total_Particles);		// vector for storing the A*P multiplication
	thrust::device_vector<T>	GPU_P	(Total_Particles);		// P vector
	// ------------------------------------------------------------------------------------------ //

	// Auxiliary values
	int  k		= 1;	// to control possible overflow of iterations
	bool flag	= true;	// to stop the iterations
	T	 sq_error;		// square error

// Initialize Conjugate gradient method ----------------------------------------------------
	// Compute A * X0
	MATRIX_VECTOR_MULTIPLICATION <T> << < Blocks, Threads >> > (raw_pointer_cast(&GPU_AUX[0]), raw_pointer_cast(&GPU_lambdas[0]), raw_pointer_cast(&GPU_Index_array[0]),
		raw_pointer_cast(&GPU_Mat_entries[0]), Total_Particles, raw_pointer_cast(&GPU_Num_Neighbors[0]), MaxNeighborNum);
	gpuError_Check(cudaDeviceSynchronize());

	// Compute R=B-A*X0
	DIFF_VECS <T> << <Blocks, Threads >> > (raw_pointer_cast(&GPU_R[0]), raw_pointer_cast(&GPU_AdaptPDF[0]), raw_pointer_cast(&GPU_AUX[0]), Total_Particles);
	gpuError_Check(cudaDeviceSynchronize());

	T Alpha, R0_norm, aux, beta;

	GPU_P = GPU_R;

	while (flag) { // this flag is useful to know when we have arrived to the desired tolerance
	// Alpha computation (EVERYTHING IS CORRECT!)
		// 1.1.- Compute AP=A*P
		MATRIX_VECTOR_MULTIPLICATION <T> << < Blocks, Threads >> > (raw_pointer_cast(&GPU_AP[0]), raw_pointer_cast(&GPU_P[0]), raw_pointer_cast(&GPU_Index_array[0]),
			raw_pointer_cast(&GPU_Mat_entries[0]), Total_Particles, raw_pointer_cast(&GPU_Num_Neighbors[0]), MaxNeighborNum);
		gpuError_Check(cudaDeviceSynchronize());

		// 1.2.- Compute P'*AP
		thrust::transform(GPU_P.begin(), GPU_P.end(), GPU_AP.begin(), GPU_AUX.begin(), thrust::multiplies<T>());
		aux = thrust::reduce(thrust::device, GPU_AUX.begin(), GPU_AUX.end());

		// 1.3.- R'*R
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_AUX.begin(), thrust::multiplies<T>());
		R0_norm = thrust::reduce(thrust::device, GPU_AUX.begin(), GPU_AUX.end());

		Alpha = R0_norm / aux;

		// New X and R: (new, old, scalar, driving vec, total length)
		// 1.- Update Lambdas
		UPDATE_VEC <T> << <Blocks, Threads >> > (raw_pointer_cast(&GPU_lambdas[0]), raw_pointer_cast(&GPU_lambdas[0]), Alpha, raw_pointer_cast(&GPU_P[0]), Total_Particles);
		gpuError_Check(cudaDeviceSynchronize());

		// 2.- Update residuals 
		UPDATE_VEC <T> << <Blocks, Threads >> > (raw_pointer_cast(&GPU_R[0]), raw_pointer_cast(&GPU_R[0]), -Alpha, raw_pointer_cast(&GPU_AP[0]), Total_Particles);
		gpuError_Check(cudaDeviceSynchronize());

		// Compute residual norm
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_AUX.begin(), thrust::multiplies<T>());
		sq_error = thrust::reduce(thrust::device, GPU_AUX.begin(), GPU_AUX.end()); // sum of its elements
		sq_error = sqrt(sq_error);

		if (sq_error < in_tolerance) {
			//std::cout << "Convergence success. Iterations:" << Iteration_information[0] << " Error:" << Iteration_information[1] << "\n";
			flag = false;
			return 0;
		}
		else if (k > max_steps) {
			std::cout << "No convergence was obtained. Last residual norm was: " << sq_error << "\n";
			std::cout << "/-------------------------------------------------------------------/\n";

			std::cin.get();
			return -1;
		}
		else {
			beta = sq_error * sq_error / R0_norm;

			UPDATE_VEC <T> << <Blocks, Threads >> > (raw_pointer_cast(&GPU_P[0]), raw_pointer_cast(&GPU_R[0]), beta, raw_pointer_cast(&GPU_P[0]), Total_Particles);
			gpuError_Check(cudaDeviceSynchronize());
			k++;
		}
	}
	return 0;
}


/// @brief 
/// @param New_GRID_PDF 
/// @param Fixed_Mesh 
/// @param Particle_Positions 
/// @param Lambdas 
/// @param Parameter_Mesh 
/// @param search_radius 
/// @param Grid_Nodes 
/// @param Particles_Per_Sample 
/// @param Total_Particles 
/// @return 
__global__ void RESTART_GRID(double*	 New_GRID_PDF,
							gridPoint*	 Fixed_Mesh,
							gridPoint*	 Particle_Positions,
							double*		 Lambdas,
							Param_vec*	 Parameter_Mesh,
							const double search_radius,
							const int	 Grid_Nodes,
							const int	 Particles_Per_Sample,
							const int	 Total_Particles) {
// OUTPUT: New values of the PDF at the fixed grid

const int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < Grid_Nodes) {
	const gridPoint Aux_FixedMesh = Fixed_Mesh[i];
	int				current_sample;
	double			dist, sum = 0;

	for (int j = 0; j < Total_Particles; j++) {
		dist = Distance(Particle_Positions[j], Aux_FixedMesh) / search_radius;	// distance between the grid point and the 

		if (dist <= 1) {
			current_sample = floorf(j / Particles_Per_Sample);					// current sample we're working with
			sum += RBF(search_radius, dist) * Lambdas[j] * Parameter_Mesh[current_sample].Joint_PDF;
		}
	}
	New_GRID_PDF[i] = cuda_fmax(sum, 0);
}
}


/// @brief This function computes the neighboring grid nodes for each advected particle from the simulation. Note that we're going to have a coo format. 
// OUTPUT IS THE COO FORMAT OF THE TRANSPOSED SPARSE MATRIX
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
									double* 		 PDF,
									double* 		 lambdas,
									const gridPoint* Fixed_Mesh,
									const Param_vec* Parameter_Mesh,
									const double 	 search_radius,
									const gridPoint  lowest_node,
									const double 	 grid_discretization_length,
									const int	 	 PtsPerDimension,
									const int	 	 Adapt_Pts,
									const int	 	 Current_sample) {
// OUTPUT: New values of the PDF at the fixed grid

const int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < Adapt_Pts) {
	unsigned int num_neighbors_per_dim 		= (unsigned int) 2 * floorf(search_radius / grid_discretization_length) + 1;
	unsigned int num_neighbors_per_particle = (unsigned int) powf(num_neighbors_per_dim, DIMENSIONS);

	gridPoint particle = Particle_Positions[i + Current_sample * Adapt_Pts];
	
	double weighted_lambda = lambdas[i + Current_sample * Adapt_Pts] * Parameter_Mesh[Current_sample].Joint_PDF;

// I want to compute the index of the lowest neighboring grid node and build its nearest neighbors
	int lowest_idx = 0;
	for (unsigned int d = 0; d < DIMENSIONS; d++){
		lowest_idx += (roundf((float) (particle.dim[d] - lowest_node.dim[d]) / grid_discretization_length) - floorf((float) search_radius / grid_discretization_length)) * powf(PtsPerDimension, d);
	}

	double dist = Distance(Fixed_Mesh[lowest_idx], particle) / search_radius;
	if (dist <= 1){
		dist = RBF(search_radius, dist) * weighted_lambda;
		_D_atomicAdd(&PDF[lowest_idx], dist);
	}

// now, go through all the neighboring grid nodes and add the values to the PDF field
	for(unsigned int j = 1; j < num_neighbors_per_particle; j++){
		int idx = lowest_idx;

		for (unsigned int d = 0; d < DIMENSIONS; d++){
			idx += (int) floorf( positive_rem(j, (int)powf(num_neighbors_per_dim, d + 1)) / powf(num_neighbors_per_dim, d) ) * powf(PtsPerDimension, d);
		}

		dist = Distance(Fixed_Mesh[idx], particle) / search_radius;
		if (dist <= 1){
			dist = RBF(search_radius, dist) * weighted_lambda;
			_D_atomicAdd(&PDF[idx], dist);
		}
	}
}
}

/// @brief 
/// @param Particle_Positions 
/// @param PDF 
/// @param lambdas 
/// @param Fixed_Mesh 
/// @param Impulse_weights 
/// @param search_radius 
/// @param lowest_node 
/// @param grid_discretization_length 
/// @param PtsPerDimension 
/// @param Adapt_Pts 
/// @param Total_Particles 
/// @return 
__global__ void RESTART_GRID_FIND_GN_II(gridPoint*  Particle_Positions,
									double* 		PDF,
									double* 		lambdas,
									const gridPoint* Fixed_Mesh,
									const double* 	Impulse_weights,
									const double 	search_radius,
									const gridPoint lowest_node,
									const double 	grid_discretization_length,
									const int	 	PtsPerDimension,
									const int	 	Adapt_Pts,
									const int	 	Current_sample) {
// OUTPUT: New values of the PDF at the fixed grid

const int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < Adapt_Pts) {
	int num_neighbors_per_dim 		= (int) 2 * floorf(search_radius / grid_discretization_length) + 1;
	int num_neighbors_per_particle 	= (int) powf(num_neighbors_per_dim, DIMENSIONS);

	gridPoint particle 	= Particle_Positions[i + Current_sample * Adapt_Pts];

	double weighted_lambda = lambdas[i + Current_sample * Adapt_Pts] * Impulse_weights[Current_sample];				// the specific sample weight

	// I want to compute the index of the lowest neighboring grid node and build its nearest neighbors
	int lowest_idx = 0;
	for (unsigned int d = 0; d < DIMENSIONS; d++){
		lowest_idx += (roundf((particle.dim[d] - lowest_node.dim[d]) / grid_discretization_length) - floorf(search_radius / grid_discretization_length)) * powf(PtsPerDimension, d);
	}

	// store the lowest sparse index identification (remember we are alredy storing the transposed matrix. The one we will need for multiplication)
	double dist = Distance(Fixed_Mesh[lowest_idx], particle) / search_radius;
	if (dist <= 1){
		dist = RBF(search_radius, dist) * weighted_lambda;
		_D_atomicAdd(&PDF[lowest_idx], dist);
	}

	// now, go through all the neighboring grid nodes and add the values to the PDF field
	for(unsigned int j = 1; j < num_neighbors_per_particle; j++){
		int idx = lowest_idx;

		for (unsigned int d = 0; d < DIMENSIONS; d++){
			idx += (int) floorf( positive_rem(j, (int)powf(num_neighbors_per_dim, d + 1)) / powf(num_neighbors_per_dim, d) ) * powf(PtsPerDimension, d);
		}

		dist = Distance(Fixed_Mesh[idx], particle) / search_radius;
		if (dist <= 1){
			dist = RBF(search_radius, dist) * weighted_lambda;
			_D_atomicAdd(&PDF[idx], dist);
		}
	}
}
}

/// @brief This function makes sure that the PDF does not have any negative values! (Having this enforced by the interpolation would've been wonderful)
/// @param PDF 
/// @param Grid_Nodes 
/// @return 
__global__ void CORRECTION(double* PDF, const int Grid_Nodes){
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Grid_Nodes){
		PDF[i] = cuda_fmax(PDF[i], 0);
	}
}

#endif