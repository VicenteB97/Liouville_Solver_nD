#include <PointSearch.cuh>

/// @brief 
/// @param Bin_locations 
/// @param Bin_count 
/// @param Bin_local_accSum 
/// @param Search_Particles 
/// @param Bounding_Box 
/// @param Total_Particles 
/// @return 
__global__ void Bin_Insertion_Count(uintType* Bin_locations,
	uintType* Bin_count,
	uintType* Bin_local_accSum,
	const Particle* Search_Particles,
	const cartesianMesh	Bounding_Box,
	const uintType	Total_Particles) {

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= Total_Particles) { return; }

	Particle temp_GP = Search_Particles[globalID];

	// Find the Bin where it belongs. We use an unsigned int because since we are working with the bounding box,
	// we're never going to have a point outside. Therefore, there will be a positive index for the bin location.

	uintType bin_idx = Bounding_Box.get_bin_idx(temp_GP);	// this part should be positive...why aren't we getting a positive number?

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
__global__ void Count_sort(const Particle* fixed_Particles,
	Particle* Particles_2sort,
	const floatType* fixed_values,
	floatType* values_2sort,
	const uintType* Bin_count,
	const uintType* Bin_locations,
	const uintType* Bin_local_accSum,
	const uintType				Total_Particles,
	const intType				offset) {

	const uintType globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= Total_Particles) { return; }

	// read bin location (@ the bounding box)
	uintType my_bin_idx = Bin_locations[globalID];

	// read the prefix sum of the bin!
	uintType pref_sum_idx = Bin_count[my_bin_idx];

	// read local cumulative sum
	uintType my_count_loc = Bin_local_accSum[globalID];

	// Add to ordered particle array (pref. sum tells the first indx of the particles to be added, where local_count)
	// tells the position among those particles
	uintType final_idx = pref_sum_idx + my_count_loc;

	Particles_2sort[final_idx] = fixed_Particles[globalID + offset];
	values_2sort[final_idx] = fixed_values[globalID + offset];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 
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
__global__ void Neighbor_search(Particle* Search_Particles,
	const uintType* Bin_count,
	intType* Index_Array,
	floatType* Matrix_Entries,
	const uintType	max_neighbor_num,
	const uintType	Total_Particles,
	const uintType	offset,
	const floatType			search_distance,			// This tells us how many discretizations we have to move back to find initial bin to search from
	const cartesianMesh		Bounding_Box) {

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// MAYBE: We could use shared memory to access particles nearby (same/near bins)!
	if (globalID >= Total_Particles) { return; }

	// Go through the corresponding bins and their particles! Note that we've got to visit some nodes in every direction
	// That includes left and right (1D) + up and down (if 2D) + front and behind (if 3D) + ...
	// But! Since we know the particles assigned in each bin and how many particles there are in each bin, we can visit efficiently!

	// Read the input from the sorted Particle array
	Particle fixed_GP = Search_Particles[globalID + offset];

	uintType temp_counter = 0;
	uintType temp_ID = (globalID + offset) * max_neighbor_num;

	// Now, we go to the lowest bin and we search all the way up!
	const uintType Bin_offset = lroundf(DISC_RADIUS);

	// We find the lowest corner in the neighboring bins (Hence, the Bin_offset variable)
	uintType bin_idx = Bounding_Box.get_bin_idx(fixed_GP, -Bin_offset);

	for (uintType k = 0; k < pow(2 * Bin_offset + 1, PHASE_SPACE_DIMENSIONS); k++) { // That's the total number of bins to visit

		// First: find the global index of the bin to search!
		intType aux_index = bin_idx;
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			aux_index += floorf(positive_rem(k, (uintType)pow(2 * Bin_offset + 1, d + 1)) / (uintType)pow(2 * Bin_offset + 1, d))
				* (uintType)pow(Bounding_Box.__nodes_per_dim, d);
		}

		if (aux_index > -1 && aux_index < Bounding_Box.total_nodes()) { // If we are selecting nodes inside the domain

			// Now that we have the bin to search, all we have to do is go through the particles that are in the corresponding bin:
			uintType startIdx = Bin_count[aux_index];
			uintType endIdx = Total_Particles;

			// When we haven't reached the last element of the count vector, we have to know where do we stop the counting
			if (aux_index < Bounding_Box.total_nodes() - 1) {
				endIdx = Bin_count[aux_index + 1];
			}

			// Add the points in the current bin
			for (uintType m = startIdx; m < endIdx; m++) {

				Particle	temp_GP = Search_Particles[m + offset];
				floatType		dist = fixed_GP.Distance(temp_GP) / search_distance;

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

// Counting sort! This operation orders the cartesianMesh indeces and it sets the number of nodes inside each bin

/// @brief 
/// @tparam floatType 
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
	thrust::device_vector<floatType>& PDF_vals,
	thrust::device_vector<intType>& Index_Array,
	thrust::device_vector<floatType>& Matrix_Entries,
	const uintType Adapt_Points,
	const uintType max_neighbor_num,
	const cartesianMesh& Bounding_Box,
	const floatType search_radius) {

	// We need our particles
	uintType Total_Particles = Search_Particles.size();

	thrust::device_vector<uintType> Bin_locations(Adapt_Points, 0);

	thrust::device_vector<Particle> temp_Particles(Adapt_Points);		// this one will store the ordered particles!
	thrust::device_vector<floatType>		temp_values(Adapt_Points, (floatType)0);		// this one will store the values of the particles in the new order!

	// Create a bounding box covering the same area as the bounding box from the particles, but change the number of nodes per dim!
	cartesianMesh CS_BBox(Bounding_Box);
	CS_BBox.__nodes_per_dim = 64;

	// Bin the particles!
	uintType Threads = fmin(THREADS_P_BLK, Adapt_Points);
	uintType Blocks = floor((Adapt_Points - 1) / Threads) + 1;

	for (uintType k = 0; k < Total_Particles / Adapt_Points; k++) { // We've got to repeat this once per random sample...we'll see another way out...

		uintType offset = k * Adapt_Points;

		// Reinitialize the particle count array
		thrust::device_vector<uintType> Bin_globalCount(CS_BBox.total_nodes(), 0);
		thrust::device_vector<uintType> Bin_localCount(Adapt_Points, 0);

		// Insert and count the particles inside each bin
		Bin_Insertion_Count << <Threads, Blocks >> > (rpc(Bin_locations, 0),
			rpc(Bin_globalCount, 0),
			rpc(Bin_localCount, 0),
			rpc(Search_Particles, offset),
			CS_BBox,
			Adapt_Points);
	if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}

		// Prefix sum of the bin-count array, which gives the cumulative points in each bin (without including the bin itself; thus the name exclusive)
		thrust::exclusive_scan(thrust::device, Bin_globalCount.begin(), Bin_globalCount.end(), Bin_globalCount.begin());

		// Counting sort...we want to sort Search_Particles!
		Count_sort << <Threads, Blocks >> > (rpc(Search_Particles, 0),
			rpc(temp_Particles, 0),
			rpc(PDF_vals, 0),
			rpc(temp_values, 0),
			rpc(Bin_globalCount, 0),
			rpc(Bin_locations, 0),
			rpc(Bin_localCount, 0),
			Adapt_Points,
			offset);
	if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}

		// Relabel the particles and their values
		thrust::copy(temp_Particles.begin(), temp_Particles.end(), &Search_Particles[offset]);
		thrust::copy(temp_values.begin(), temp_values.end(), &PDF_vals[offset]);

		// Find particle neighbors using all the previous information
		Neighbor_search << <Threads, Blocks >> > (rpc(Search_Particles, 0),
			rpc(Bin_globalCount, 0),
			rpc(Index_Array, 0),
			rpc(Matrix_Entries, 0),
			max_neighbor_num,
			Adapt_Points,
			offset,
			search_radius,
			CS_BBox);
	if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}
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
/// @tparam floatType
/// @param Search_Particles 
/// @param Fixed_Particles 
/// @param Index_Array 
/// @param Matrix_Entries
/// @param max_neighbor_num 
/// @param Adapt_Points 
/// @param Total_Particles 
/// @param search_radius 
/// @return 
__global__ void Exh_PP_Search(
	const Particle* Search_Particles,
	const Particle* Fixed_Particles,
	intType* Index_Array,
	floatType* Matrix_Entries,
	const intType max_neighbor_num,
	const uintType Adapt_Points,
	const uintType Total_Particles,
	const floatType search_radius) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= Total_Particles) { return; }

	Particle	FP_aux = Fixed_Particles[i];									// Tells me what parameter sample I'm at
	const intType	k = i * max_neighbor_num;

	Index_Array[k] = i;
	Matrix_Entries[k] = RBF(search_radius, 0);

	uintType		aux = 1;
	const intType	i_aux = (double)floor((double)i / Adapt_Points);	// this double is used so that the compiler knows that we are using the CUDA version
	floatType			dist;

	for (uintType j = i_aux * Adapt_Points; j < (i_aux + 1) * Adapt_Points; j++) {		// neighborhood where I'm searching

		Particle Temp_Particle = Search_Particles[j];

		dist = Temp_Particle.Distance(FP_aux) / search_radius;	// normalized distance between particles

		if (dist <= 1 && aux < max_neighbor_num && j != i) {
			Index_Array[k + aux] = j;
			Matrix_Entries[k + aux] = RBF(search_radius, dist);
			aux++;
		}
	}
}


int16_t particleNeighborSearch(
	thrust::device_vector<Particle>& Search_Particles,
	thrust::device_vector<floatType>& PDF_vals,
	thrust::device_vector<intType>& Index_Array,
	thrust::device_vector<floatType>& Matrix_Entries,
	const uintType Adapt_Points,
	const uintType MaxNeighborNum,
	const cartesianMesh& Bounding_Box,
	const floatType search_radius) {

	uint16_t	Threads = fmin(THREADS_P_BLK, Adapt_Points);
	uintType		Blocks = floor((Adapt_Points - 1) / Threads) + 1;

	Exh_PP_Search << <Blocks, Threads >> > (rpc(Search_Particles, 0),
		rpc(Search_Particles, 0),
		rpc(Index_Array, 0),
		rpc(Matrix_Entries, 0),
		MaxNeighborNum,
		Adapt_Points,
		Adapt_Points,
		search_radius);
if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}

	// Dynamical choice of either exhaustive or counting sort-based point search
	if (Adapt_Points < ptSEARCH_THRESHOLD) {
		Exh_PP_Search << <Blocks, Threads >> > (rpc(Search_Particles, 0),
			rpc(Search_Particles, 0),
			rpc(Index_Array, 0),
			rpc(Matrix_Entries, 0),
			MaxNeighborNum,
			Adapt_Points,
			Adapt_Points,
			search_radius);
	if(cudaDeviceSynchronize()!=cudaSuccess){return EXIT_FAILURE;}
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
