#include "Domain.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class definition
	// Parametric constructors:
	/// @brief Create a cartesianMesh knowing the nodes per dimension
	/// @param nodes_per_dim 
	/// @return 
hostFunction deviceFunction	cartesianMesh::cartesianMesh(intType nodes_per_dim) :
	__nodes_per_dim(nodes_per_dim),
	__boundary_inf(Particle(DOMAIN_INF)),
	__boundary_sup(Particle(DOMAIN_SUP))
{};

/// @brief Create a cartesianMesh knowing the discretization length
/// @param discretization_length 
/// @return 
hostFunction deviceFunction	cartesianMesh::cartesianMesh(floatType discretization_length) :
	__boundary_inf(Particle(DOMAIN_INF)),
	__boundary_sup(Particle(DOMAIN_SUP))
{
	__nodes_per_dim = roundf((floatType)(__boundary_sup.dim[0] - __boundary_inf.dim[0]) / discretization_length);
};

/// @brief Create a cartesianMesh specifying all the parameters
/// @param bnd_inf 
/// @param bnd_sup 
/// @param nodes_per_dim 
/// @return 
hostFunction deviceFunction	cartesianMesh::cartesianMesh(const Particle& bnd_inf, const Particle& bnd_sup, intType nodes_per_dim) :
	__nodes_per_dim(nodes_per_dim),
	__boundary_inf(bnd_inf),
	__boundary_sup(bnd_sup)
{};

/// @brief Create a cartesianMesh specifying all the parameters (discr. length instead of the nodes per dimension)
/// @param bnd_inf 
/// @param bnd_sup 
/// @param discretization_length 
/// @return 
hostFunction deviceFunction	cartesianMesh::cartesianMesh(const Particle& bnd_inf, const Particle& bnd_sup, floatType discretization_length) :
	__boundary_inf(bnd_inf),
	__boundary_sup(bnd_sup)
{
	__nodes_per_dim = (__boundary_sup.dim[0] - __boundary_inf.dim[0]) / discretization_length + 1;
};

hostFunction deviceFunction
cartesianMesh::cartesianMesh(const cartesianMesh& otherMesh) :
	__nodes_per_dim(otherMesh.nodes_per_dim()),
	__boundary_inf(otherMesh.boundary_inf()),
	__boundary_sup(otherMesh.boundary_sup())
{};

// Methods/functions
hostFunction deviceFunction
void cartesianMesh::setBoundaryInf(const Particle& boundary_inf) {
	__boundary_inf = boundary_inf;
};

hostFunction deviceFunction
Particle cartesianMesh::boundary_inf() const {
	return __boundary_inf;
};

hostFunction deviceFunction
void cartesianMesh::setBoundarySup(const Particle& boundary_sup) {
	__boundary_sup = boundary_sup;
};

hostFunction deviceFunction
Particle cartesianMesh::boundary_sup() const {
	return __boundary_sup;
};

hostFunction deviceFunction
void cartesianMesh::setNodesPerDimension(uintType nodes_per_dim) {
	__nodes_per_dim = nodes_per_dim;
};

hostFunction deviceFunction
uintType cartesianMesh::nodes_per_dim() const {
	return __nodes_per_dim;
};

	/// @brief Compute the total number of nodes
hostFunction deviceFunction	 
intType cartesianMesh::total_nodes() const {
	return pow(__nodes_per_dim, PHASE_SPACE_DIMENSIONS);
}

/// @brief Gives the edge length (side length of a cube)
hostFunction deviceFunction	 
floatType cartesianMesh::edge_size() const {
	return (__boundary_sup.dim[0] - __boundary_inf.dim[0]);
}

/// @brief Gives the center node of the cartesianMesh
hostFunction deviceFunction	 
Particle cartesianMesh::center() const {
	return (__boundary_sup + __boundary_inf)*(0.5);
}

/// @brief Gives the discretization length (distance between two consecutive nodes in the same dimension) 
hostFunction deviceFunction	 
floatType cartesianMesh::discr_length() const {
	if (__nodes_per_dim == 1) { return (floatType)0; }

	return (floatType)(this->edge_size() / (__nodes_per_dim - 1));
}

/// @brief Gives the node (point in space) given the global index in the cartesianMesh
/// @param globalIdx Global index in the current cartesianMesh
/// @return point in space
hostFunction deviceFunction	 
Particle cartesianMesh::get_node(uint64_t globalIdx) const {

	Particle out(__boundary_inf);
	intType temp = 1;
	floatType discretizationLength = this->discr_length();

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		uint64_t j = floorf((floatType)positive_rem(globalIdx, temp * __nodes_per_dim) / temp);	// This line gives the index at each dimension

		out.dim[d] += j * discretizationLength; temp *= __nodes_per_dim;			// This line gives the cartesianMesh node per se
	}
	return out;
}

/// @brief This method decides whether Particle is inside the cartesianMesh or not
/// @param Particle 
/// @return bool. True if particle is inside cartesianMesh, false otherwise
hostFunction deviceFunction	 
bool cartesianMesh::containsParticle(const Particle& particle) const {
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		if (particle.dim[d] < __boundary_inf.dim[d] || particle.dim[d] > __boundary_sup.dim[d]) { return false; }
	}
	return true;
}

// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
hostFunction deviceFunction  
uint64_t cartesianMesh::getBinIdx(const Particle& particle, intType bin_offset) const {
	int64_t bin_idx = 0, accPower = 1;
	floatType discretizationLength = this->discr_length();

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		int64_t temp_idx = roundf((floatType)(particle.dim[d] - __boundary_inf.dim[d]) / discretizationLength) + bin_offset;

		bin_idx += temp_idx * accPower;
		accPower *= __nodes_per_dim;
	}
	return bin_idx;
};

// Compute the global index at your mesh, given the global index in "other" mesh.
hostFunction deviceFunction
uint64_t cartesianMesh::idx_here_from_other_mesh(intType indx_at_other, const cartesianMesh& other) const {
	return this->getBinIdx(other.get_node(indx_at_other));
}

/// @brief This function expands a fixed cartesianMesh "other" by a length of  "expansion_length" in each direction/dimension
/// @param other The base cartesianMesh from which we will expand
/// @param expansion_nodes Number of nodes we will expand in every direction
hostFunction deviceFunction	 
void cartesianMesh::Expand_From(const cartesianMesh& other, intType expansion_nodes) {

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		// To make sure that the points fall into the cartesianMesh nodes
		__boundary_inf.dim[d] = other.boundary_inf().dim[d] - other.discr_length() * expansion_nodes;
		__boundary_sup.dim[d] = other.boundary_sup().dim[d] + other.discr_length() * expansion_nodes;
	}

	__nodes_per_dim = other.nodes_per_dim() + 2 * expansion_nodes;
}

/// @brief This function makes you domain a square (same Lebesgue-length in every direction)
hostFunction deviceFunction  
void cartesianMesh::Squarify() {
	// Get the max distance between the edges and then make the box larger!
	floatType max_dist = __boundary_sup.dim[0] - __boundary_inf.dim[0];

	for (uint16_t d = 1; d < PHASE_SPACE_DIMENSIONS; d++) {
		max_dist = fmaxf((floatType)max_dist, __boundary_sup.dim[d] - __boundary_inf.dim[d]);
	}

	// Now that we know the max dist, let's expand the edges!
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		__boundary_sup.dim[d] = __boundary_inf.dim[d] + max_dist;
	}
}

/// @brief This function updates a cartesianMesh-defined bounding box
/// @param fullParticleLocations_dvc GPU array storing the positions of the particles
/// @returns Nothing
void cartesianMesh::update_bounding_box(const deviceUniquePtr<Particle>& fullParticleLocations_dvc) {

	uint64_t particleCount = fullParticleLocations_dvc.size_count();
	intType threads = fmin(THREADS_P_BLK, particleCount);
	intType blocks = floor((particleCount - 1) / threads) + 1;

	// Temporary vector storing the particles' projections in each dimension 
	//deviceUniquePtr<floatType> projection(fullParticleLocations_dvc.size(), (floatType)0);
#ifdef USECUDA
	deviceUniquePtr<floatType> projection(particleCount, (floatType)0);
	gpu_device.memCpy_dvc2dvc(
		projection.get(),
		fullParticleLocations_dvc.get(),
		fullParticleLocations_dvc.size_bytes()
	);
	thrust::device_ptr<floatType> projectionPtr(projection.get());
#endif

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {

		gpu_device.launchKernel(blocks, threads, find_projection{
			fullParticleLocations_dvc.get(),
			projection.get(),
			particleCount,
			d
		});

		floatType temp_1 = *(thrust::min_element(thrust::device, projectionPtr, projectionPtr + particleCount)); // min element from the projection in that direction
		floatType temp_2 = *(thrust::max_element(thrust::device, projectionPtr, projectionPtr + particleCount));

		__boundary_inf.dim[d] = temp_1 - ceilf(DISC_RADIUS) * this->discr_length();
		__boundary_sup.dim[d] = temp_2 + ceilf(DISC_RADIUS) * this->discr_length();
	}
}
