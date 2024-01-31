#ifndef __DOMAIN_CUH__
#define __DOMAIN_CUH__

#include "Constants.cuh"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid points ---------------------------------------------------------------------------------------------------------------- //

class Particle {
public:
	TYPE dim[PHASE_SPACE_DIMENSIONS];

	// Default constructor
	__host__ __device__
	Particle() {
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			dim[d] = (TYPE)0;
		}
	}

	// Parametric constructors
	__host__ __device__
		Particle(const TYPE(&input)[PHASE_SPACE_DIMENSIONS]) {
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			dim[d] = input[d];
		}
	}

	__host__ __device__
	Particle(const Particle& input) {
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			dim[d] = input.dim[d];
		}
	}

public:
	// Operators and methods
	__host__ __device__ 
	Particle operator+(const Particle& other) const {

		Particle out;

		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			out.dim[d] = dim[d] + other.dim[d];
		}

		return out;
	}
	__host__ __device__ 
	Particle operator-(const Particle& other) const {
		Particle out;

		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			out.dim[d] = dim[d] - other.dim[d];
		}

		return out;
	}
	__host__ __device__ 
	bool operator==(const Particle& other) const {
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			if (dim[d] != other.dim[d]) { return false; }
		}
		return true;
	}

	__host__ __device__ 
	inline TYPE Distance(const Particle& other) const {
		TYPE dist = 0;
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			dist += (dim[d] - other.dim[d]) * (dim[d] - other.dim[d]);
		}
		return sqrtf(dist);
	}

	
	__host__ __device__ 
	inline Particle Mult_by_Scalar(TYPE scalar) const {
		Particle out;
		
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			out.dim[d] = scalar * dim[d];
		}

		return out;
	}
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class ----------------------------------------------------------------------------------------------------------------- //

// This function is defined aside because CUDA does not allow defining __global__ functions inside class definitions! (At least not statically)
 
__global__ void findProjection(const Particle* particles, TYPE* projections, INT totalParticles, INT dimension) {
	
	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= totalParticles) { return; }

	projections[globalID] = particles[globalID].dim[dimension];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class definition

class Mesh{
public:
	Particle 	Boundary_inf, Boundary_sup;
	INT		Nodes_per_Dim;

	// Parametric constructors:
	/// @brief Create a Mesh knowing the nodes per dimension
	/// @param Nodes_per_dim 
	/// @return 
	__host__ __device__	Mesh(INT Nodes_per_dim = 2) {

		Nodes_per_Dim = Nodes_per_dim;
		Boundary_inf  = Particle(DOMAIN_INF);
		Boundary_sup  = Particle(DOMAIN_SUP);
	}

	/// @brief Create a Mesh knowing the discretization length
	/// @param Discretization_length 
	/// @return 
	__host__ __device__	Mesh(TYPE Discretization_length) {

		Boundary_inf = Particle(DOMAIN_INF);
		Boundary_sup = Particle(DOMAIN_SUP);
		Nodes_per_Dim = roundf((TYPE)(Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Discretization_length);
	}

	/// @brief Create a Mesh specifying all the parameters
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @param Nodes_per_dim 
	/// @return 
	__host__ __device__	Mesh(const Particle& Bnd_inf, const Particle& Bnd_sup, INT Nodes_per_dim = 2){

		Nodes_per_Dim	= Nodes_per_dim;
		Boundary_inf	= Bnd_inf;
		Boundary_sup	= Bnd_sup;
	}

	/// @brief Create a Mesh specifying all the parameters (discr. length instead of the nodes per dimension)
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @param Discretization_length 
	/// @return 
	__host__ __device__	Mesh(const Particle& Bnd_inf, const Particle& Bnd_sup, TYPE Discretization_length) {

		Boundary_inf = Bnd_inf;
		Boundary_sup = Bnd_sup;
		Nodes_per_Dim = (Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Discretization_length + 1;
	}

// Methods/functions
public:
	/// @brief Compute the total number of nodes
	__host__ __device__	inline INT Total_Nodes() const {
		return pow(Nodes_per_Dim, PHASE_SPACE_DIMENSIONS);
	}

	/// @brief Gives the edge length (side length of a cube)
	__host__ __device__	inline TYPE Edge_size() const {
		return (Boundary_sup.dim[0] - Boundary_inf.dim[0]);
	}

	/// @brief Gives the center node of the Mesh
	__host__ __device__	inline Particle Center() const {
		return (Boundary_sup + Boundary_inf).Mult_by_Scalar(0.5);
	}

	/// @brief Gives the discretization length (distance between two consecutive nodes in the same dimension) 
	__host__ __device__	inline TYPE Discr_length() const {
		if (Nodes_per_Dim == 1) { return (TYPE)0; }

		return (TYPE)(this->Edge_size() / (Nodes_per_Dim - 1));
	}

	/// @brief Gives the node (point in space) given the global index in the Mesh
	/// @param globalIdx Global index in the current Mesh
	/// @return point in space
	__host__ __device__	inline Particle Get_node(INT globalIdx) const {

		Particle out(Boundary_inf);
		INT temp = 1;
		TYPE discretizationLength = this->Discr_length();

		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT j = floorf((TYPE) positive_rem(globalIdx, temp * Nodes_per_Dim) / temp );	// This line gives the index at each dimension

			out.dim[d] += j * discretizationLength; temp *= Nodes_per_Dim;			// This line gives the Mesh node per se
		}
		return out;
	}

	/// @brief This method decides whether Particle is inside the Mesh or not
	/// @param Particle 
	/// @return bool. True if particle is inside Mesh, false otherwise
	__host__ __device__	inline bool Contains_particle(const Particle& Particle) const {
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			if (Particle.dim[d] < Boundary_inf.dim[d] || Particle.dim[d] > Boundary_sup.dim[d]) { return false; }
		}
		return true;
	}
	
	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	__host__ __device__ inline INT Get_binIdx(const Particle& Particle, INT bin_offset = 0) const {
		INT bin_idx = 0, accPower = 1;
		TYPE discretizationLength = this->Discr_length();
 
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT temp_idx = roundf((TYPE)(Particle.dim[d] - Boundary_inf.dim[d]) / discretizationLength) + bin_offset;

			bin_idx  += temp_idx * accPower;
			accPower *= Nodes_per_Dim;
		}
		return bin_idx;
	};

	// Compute the global index at your mesh, given the global index in "other" mesh.
	__host__ inline INT Indx_here(INT indx_at_other, const Mesh& other) const {
		return this->Get_binIdx(other.Get_node(indx_at_other));
	}

	/// @brief This function expands a fixed Mesh "Other" by a length of  "expansion_length" in each direction/dimension
	/// @param Other The base Mesh from which we will expand
	/// @param expansion_nodes Number of nodes we will expand in every direction
	__host__ __device__	inline void Expand_From(const Mesh& Other, INT expansion_nodes) {
		
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			// To make sure that the points fall into the Mesh nodes
			Boundary_inf.dim[d] = Other.Boundary_inf.dim[d] - Other.Discr_length() * expansion_nodes;
			Boundary_sup.dim[d] = Other.Boundary_sup.dim[d] + Other.Discr_length() * expansion_nodes;
		}

		Nodes_per_Dim = Other.Nodes_per_Dim + 2 * expansion_nodes;
	}

	/// @brief This function makes you domain a square (same Lebesgue-length in every direction)
	__host__ __device__ inline void Squarify() {
		// Get the max distance between the edges and then make the box larger!
		TYPE max_dist = Boundary_sup.dim[0] - Boundary_inf.dim[0];

		for (uint16_t d = 1; d < PHASE_SPACE_DIMENSIONS; d++) {
			max_dist = fmaxf((TYPE)max_dist, Boundary_sup.dim[d] - Boundary_inf.dim[d]);
		}

		// Now that we know the max dist, let's expand the edges!
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			Boundary_sup.dim[d] = Boundary_inf.dim[d] + max_dist;
		}
	}

	/// @brief This function updates a Mesh-defined bounding box
	/// @param D_Particle_Locations GPU array storing the positions of the particles
	/// @returns Nothing
	inline void Update_boundingBox(const thrust::device_vector<Particle>& D_Particle_Locations){
		
		INT Threads = fmin(THREADS_P_BLK, D_Particle_Locations.size());
		INT Blocks	 = floor((D_Particle_Locations.size() - 1) / Threads) + 1;

		// Temporary vector storing the particles' projections in each dimension 
		thrust::device_vector<TYPE> projection(D_Particle_Locations.size(),(TYPE)0);

		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			findProjection << <Blocks, Threads >> > (rpc(D_Particle_Locations, 0), rpc(projection, 0), D_Particle_Locations.size(), d);

			TYPE temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end())); // min element from the projection in that direction
			TYPE temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));
			
			Boundary_inf.dim[d] = temp_1 - ceilf(DISC_RADIUS) * this->Discr_length();
			Boundary_sup.dim[d] = temp_2 + ceilf(DISC_RADIUS) * this->Discr_length();
		}
		projection.clear();
	}
};



#endif