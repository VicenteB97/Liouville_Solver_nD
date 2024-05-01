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

#include <./include/Interpolation.cuh>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void UPDATE_VEC(floatType* x, const floatType* x0, const floatType scalar, const floatType* v, const intType Max_Length) {
	const uint64_t globalIDX = blockDim.x * blockIdx.x + threadIdx.x;

	uint64_t i = globalIDX * ELEMENTS_AT_A_TIME;

#pragma unroll
	for (uint16_t k = 0; k < ELEMENTS_AT_A_TIME; k++) {
		if ((i + k) < Max_Length) { x[i + k] = x0[i + k] + scalar * v[i + k]; }
		else { return; }
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void MATRIX_VECTOR_MULTIPLICATION(floatType* X, const floatType* x0, const intType* Matrix_idxs, const floatType* Matrix_entries, const intType total_length, const intType Max_Neighbors) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;	// For each i, which represents the matrix row, we read the index positions and multiply against the particle weights

	if (i >= total_length) { return; }						// total length = adapt_points * totalSampleCount

	// 1.- Compute A*X0										
		// 1.1.- Determine where my particles are!!
	const uintType i0 = i * Max_Neighbors;		// where does my search index start

	floatType a = 0;					// auxiliary value for sum (the diagonal is always 1 in our case)
	uintType j = i0;
	while (Matrix_idxs[j] != -1 && j < total_length * Max_Neighbors) {
		intType p = Matrix_idxs[j];

		a += Matrix_entries[j] * x0[p]; 	// < n calls to global memory
		j++;
	}

	// 2.- Output
	X[i] = a;								// particle weights
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ InterpHandle::InterpHandle(uintType size) {
	GPU_R.resize(size);
	GPU_temp.resize(size);
	GPU_AP.resize(size);
	GPU_P.resize(size);
}


void InterpHandle::resize(uintType size) {
	GPU_R.resize(size);
	GPU_temp.resize(size);
	GPU_AP.resize(size);
	GPU_P.resize(size);
};


__host__ intType CONJUGATE_GRADIENT_SOLVE(
	thrust::device_vector<floatType>& GPU_lambdas,
	thrust::device_vector<intType>& GPU_Index_array,
	thrust::device_vector<floatType>& GPU_Mat_entries,
	thrust::device_vector<floatType>& GPU_AdaptPDF,
	InterpHandle& interpVectors,
	const intType					Total_Particles,
	const intType					MaxNeighborNum,
	const uintType					max_steps,
	const floatType							in_tolerance) {


	// Determine threads and blocks for the simulation
	const uintType Threads = (uintType)fminf(THREADS_P_BLK, Total_Particles);
	const uintType Blocks = (uintType)floorf((Total_Particles - 1) / Threads) + 1;

	// These are for the update_vec function
	const uintType Threads_2 = (uintType)fminf(THREADS_P_BLK, (float)Total_Particles / ELEMENTS_AT_A_TIME);
	const uintType Blocks_2 = (uintType)floorf((float)(Total_Particles / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;

	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------------------- //
	// Auxiliary values
	intType  k = 1;	// to control possible overflow of iterations
	bool flag = true;	// to stop the iterations
	const double squaredTolerance = in_tolerance * in_tolerance;

	// Initialize Conjugate gradient method ----------------------------------------------------
		// Compute A * X0
	MATRIX_VECTOR_MULTIPLICATION << < Blocks, Threads >> > (rpc(interpVectors.GPU_temp, 0), rpc(GPU_lambdas, 0), rpc(GPU_Index_array, 0),
		rpc(GPU_Mat_entries, 0), Total_Particles, MaxNeighborNum);
	gpuError_Check(cudaDeviceSynchronize());

	// Compute R=B-A*X0
	UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(interpVectors.GPU_R, 0), rpc(GPU_AdaptPDF, 0), (floatType)-1, rpc(interpVectors.GPU_temp, 0), Total_Particles);
	gpuError_Check(cudaDeviceSynchronize());

	floatType Alpha, R0_norm, r_squaredNorm, aux, beta;
	// floatType Alpha, R0_norm, r_norm, aux, beta;

	interpVectors.GPU_P = interpVectors.GPU_R;

	while (flag) { // this flag is useful to know when we have arrived to the desired tolerance
		// Alpha computation (EVERYTHING IS CORRECT!)
			// 1.1.- Compute AP=A*P
		MATRIX_VECTOR_MULTIPLICATION << < Blocks, Threads >> > (rpc(interpVectors.GPU_AP, 0), rpc(interpVectors.GPU_P, 0), rpc(GPU_Index_array, 0),
			rpc(GPU_Mat_entries, 0), Total_Particles, MaxNeighborNum);
		gpuError_Check(cudaDeviceSynchronize());

		// 1.2.- Compute P'*AP
		aux = thrust::inner_product(thrust::device, interpVectors.GPU_P.begin(), interpVectors.GPU_P.end(), interpVectors.GPU_AP.begin(), 0.0f);

		// 1.3.- R'*R
		R0_norm = thrust::inner_product(thrust::device, interpVectors.GPU_R.begin(), interpVectors.GPU_R.end(), interpVectors.GPU_R.begin(), 0.0f);

		Alpha = R0_norm / aux;

		// New X and R: (new, old, scalar, driving vec, total length)
		// 1.- Update Lambdas
		UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(GPU_lambdas, 0), rpc(GPU_lambdas, 0), Alpha, rpc(interpVectors.GPU_P, 0), Total_Particles);
		// we DO NOT use cudaDeviceSynchronize() because the following CUDA kernel does not require this kernel to be done...we may save a (very small) amount of time

		// 2.- Update residuals 
		UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(interpVectors.GPU_R, 0), rpc(interpVectors.GPU_R, 0), -Alpha, rpc(interpVectors.GPU_AP, 0), Total_Particles);
		gpuError_Check(cudaDeviceSynchronize());

		// Compute residual l_2 norm
		r_squaredNorm = thrust::inner_product(thrust::device, interpVectors.GPU_R.begin(), interpVectors.GPU_R.end(), interpVectors.GPU_R.begin(), 0.0f);

		if ((double)r_squaredNorm / (Total_Particles * Total_Particles) < squaredTolerance) {
			flag = false;
			break;
		}
		else if (k > max_steps) {
			std::cout << "No convergence was obtained after reaching max. allowed iterations. Last residual norm was: " << sqrt(r_squaredNorm) << "\n";
			std::cout << border_mid;

			k = -1;
			flag = false;
			break;
		}
		else {
			beta = r_squaredNorm / R0_norm;

			UPDATE_VEC << <Blocks_2, Threads_2 >> > (rpc(interpVectors.GPU_P, 0), rpc(interpVectors.GPU_R, 0), beta, rpc(interpVectors.GPU_P, 0), Total_Particles);
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

__global__ void RESTART_GRID_FIND_GN(Particle* Particle_Positions,
	floatType* PDF,
	floatType* lambdas,
	const Param_pair* Parameter_Mesh,
	const intType* n_Samples,
	const floatType 	 		search_radius,
	const uintType	 		Adapt_Pts,
	const uintType	 		Block_samples,
	const uintType	 		offset,
	const Mesh 	Domain,
	const Mesh	Expanded_Domain) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Adapt_Pts * Block_samples) { return; }

	uintType Current_sample = offset + (uintType)floorf((float)i / Adapt_Pts);
	Param_vec<PARAM_SPACE_DIMENSIONS>	aux = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(Current_sample, Parameter_Mesh, n_Samples);

	floatType weighted_lambda = lambdas[i] * aux.Joint_PDF;

	Particle particle(Particle_Positions[i]);

	// Find the point in the lowest corner of the search box!
	Particle Lowest_node(Expanded_Domain.Get_node(Expanded_Domain.Get_binIdx(particle, -lround(DISC_RADIUS))));

	const uintType Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;
	const uintType totalNeighborsToVisit = pow(Neighbors_per_dim, PHASE_SPACE_DIMENSIONS);
	const floatType domainDiscretization = Domain.Discr_length();

	// Go through all the nodes where rewriting will be possible
	for (uint16_t k = 0; k < totalNeighborsToVisit; k++) {

		Particle visit_node(Lowest_node);

		// Get the node at that point
		uintType tempPowerAccumulate = 1;

#pragma unroll
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			uintType temp_idx = floorf((float) positive_rem(k, Neighbors_per_dim * tempPowerAccumulate) / tempPowerAccumulate);

			visit_node.dim[d] += temp_idx * domainDiscretization;
			tempPowerAccumulate *= Neighbors_per_dim;
		}

		// If it is inside our problem mesh...
		if (Domain.Contains_particle(visit_node)) {

			// Calculate normalized distance
			floatType dist = visit_node.Distance(particle) / search_radius;

			// if it's inside the RBF support...
			if (dist <= 1) {

				dist = RBF(search_radius, dist) * weighted_lambda;

				intType idx = Domain.Get_binIdx(visit_node);

				atomicAdd(&PDF[idx], dist);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void RESTART_GRID_FIND_GN(Particle* Particle_Positions,
	floatType* PDF,
	floatType* lambdas,
	const Param_vec<PHASE_SPACE_DIMENSIONS>* Impulse_weights,
	const floatType 	search_radius,
	const uintType	Adapt_Pts,
	const uintType	Current_sample,
	const Mesh	Domain,
	const Mesh	Expanded_Domain) {
	// OUTPUT: New values of the PDF at the fixed Mesh

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Adapt_Pts) { return; }

	uintType global_id = i + Current_sample * Adapt_Pts;

	floatType weighted_lambda = lambdas[global_id] * Impulse_weights[Current_sample].Joint_PDF;				// the specific sample weight

	Particle particle(Particle_Positions[global_id]);

	// Find the point in the lowest corner of the search box!
	Particle Lowest_node(Expanded_Domain.Get_node(Expanded_Domain.Get_binIdx(particle, -lround(DISC_RADIUS))));

	const uintType Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;
	const uintType totalNeighborsToVisit = pow(Neighbors_per_dim, PHASE_SPACE_DIMENSIONS);
	const floatType domainDiscretization = Domain.Discr_length();

	// Go through all the nodes where rewriting will be possible
	for (uint16_t k = 0; k < totalNeighborsToVisit; k++) {

		Particle visit_node(Lowest_node);

		// Get the node at that point
		uintType tempPowerAccumulate = 1;

#pragma unroll
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			uintType temp_idx = floorf(positive_rem(k, Neighbors_per_dim * tempPowerAccumulate) / tempPowerAccumulate);

			visit_node.dim[d] += temp_idx * domainDiscretization;
			tempPowerAccumulate *= Neighbors_per_dim;
		}

		// If it is inside our problem mesh...
		if (Domain.Contains_particle(visit_node)) {

			// Calculate normalized distance
			floatType dist = visit_node.Distance(particle) / search_radius;

			// if it's inside the RBF support...
			if (dist <= 1) {

				dist = RBF(search_radius, dist) * weighted_lambda;

				intType idx = Domain.Get_binIdx(visit_node);

				atomicAdd(&PDF[idx], dist);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief This function makes sure that the PDF does not have any negative values! (Having this enforced by the interpolation would've been wonderful)
/// @param PDF 
/// @param Grid_Nodes 
/// @return 

__global__ void CORRECTION(floatType* PDF, const intType Grid_Nodes) {
	const uint64_t globalIDX = blockDim.x * blockIdx.x + threadIdx.x;

	const uint64_t i = globalIDX * ELEMENTS_AT_A_TIME;

#pragma unroll
	for (uint16_t k = 0; k < ELEMENTS_AT_A_TIME; k++) {
		if (i + k < Grid_Nodes) {
			PDF[i + k] = fmaxf(PDF[i + k], 0.00f);
		}
	}


}
