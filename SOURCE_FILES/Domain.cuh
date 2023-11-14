#ifndef __DOMAIN_CUH__
#define __DOMAIN_CUH__

#include "Constants.cuh"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid points ---------------------------------------------------------------------------------------------------------------- //
template<uint16_t PHASE_SPACE_DIM, class T>
class gridPoint {
public:
	T dim[PHASE_SPACE_DIM];

	// Default constructor
	__host__ __device__
	gridPoint<PHASE_SPACE_DIM, T>() {
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			dim[d] = (T)0;
		}
	}

	// Parametric constructors
	__host__ __device__
		gridPoint<PHASE_SPACE_DIM, T>(const T(&input)[PHASE_SPACE_DIM]) {
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			dim[d] = input[d];
		}
	}

	__host__ __device__
	gridPoint<PHASE_SPACE_DIM, T>(const gridPoint<PHASE_SPACE_DIM,T>& input) {
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			dim[d] = input.dim[d];
		}
	}

	// Operators and methods
	__host__ __device__ 
	gridPoint<PHASE_SPACE_DIM, T> operator+(const gridPoint<PHASE_SPACE_DIM, T>& other) const {

		gridPoint<PHASE_SPACE_DIM, T> out;

		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			T aux = dim[d];
			aux += other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ 
	gridPoint<PHASE_SPACE_DIM, T> operator-(const gridPoint<PHASE_SPACE_DIM, T>& other) const {
		gridPoint<PHASE_SPACE_DIM, T> out;

		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			T aux = dim[d];
			aux -= other.dim[d];
			out.dim[d] = aux;
		}

		return out;
	}
	__host__ __device__ 
	bool operator==(const gridPoint<PHASE_SPACE_DIM, T>& other) const {
		bool out = true;
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			if (dim[d] != other.dim[d]) { out = false; }
		}
		return out;
	}

	__host__ __device__ 
	inline T Distance(const gridPoint<PHASE_SPACE_DIM, T>& other) const {
		T dist = 0;
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			dist += (dim[d] - other.dim[d]) * (dim[d] - other.dim[d]);
		}
		return sqrtf(dist);
	}

	
	__host__ __device__ 
	inline gridPoint<PHASE_SPACE_DIM, T> Mult_by_Scalar(T scalar) const {
		gridPoint<PHASE_SPACE_DIM, T> out;

		
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
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
template<uint16_t PHASE_SPACE_DIM, class T> 
__global__ void findProjection(const gridPoint<PHASE_SPACE_DIM, T>* particles, T* projections, const UINT totalParticles, const UINT dimension) {
	
	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalID >= totalParticles) { return; }

	projections[globalID] = particles[globalID].dim[dimension];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class definition
template<uint16_t PHASE_SPACE_DIM, class T>
class grid{
public:
	gridPoint<PHASE_SPACE_DIM, T> 	Boundary_inf, Boundary_sup;
	UINT				Nodes_per_Dim;

	// Default constructor:
	__host__ __device__	grid<PHASE_SPACE_DIM, T>(){
		Nodes_per_Dim	= 1;
		Boundary_inf	= gridPoint<PHASE_SPACE_DIM, T>(DOMAIN_INF);
		Boundary_sup	= gridPoint<PHASE_SPACE_DIM, T>(DOMAIN_SUP);
	}

	// Parametric constructors:
	
	/// @brief Create a grid knowing the nodes per dimension
	/// @param Nodes_per_dim 
	/// @return 
	__host__ __device__	grid<PHASE_SPACE_DIM, T>(const INT& Nodes_per_dim) {

		Nodes_per_Dim = Nodes_per_dim;
		Boundary_inf  = gridPoint<PHASE_SPACE_DIM, T>(DOMAIN_INF);
		Boundary_sup  = gridPoint<PHASE_SPACE_DIM, T>(DOMAIN_SUP);
	}

	/// @brief Create a grid knowing the discretization length
	/// @param Discretization_length 
	/// @return 
	__host__ __device__	grid<PHASE_SPACE_DIM, T>(const T& Discretization_length) {

		Boundary_inf = gridPoint<PHASE_SPACE_DIM, T>(DOMAIN_INF);
		Boundary_sup = gridPoint<PHASE_SPACE_DIM, T>(DOMAIN_SUP);
		Nodes_per_Dim = roundf((Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Discretization_length);
	}

	/// @brief Create a grid specifying the extremal nodes. 2 nodes per dimension by default
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @return 
	__host__ __device__	grid<PHASE_SPACE_DIM, T>(const gridPoint<PHASE_SPACE_DIM, T>& Bnd_inf, const gridPoint<PHASE_SPACE_DIM, T>& Bnd_sup) {

		Nodes_per_Dim = 2;
		Boundary_inf = Bnd_inf;
		Boundary_sup = Bnd_sup;
	}

	/// @brief Create a grid specifying all the parameters
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @param Nodes_per_dim 
	/// @return 
	__host__ __device__	grid<PHASE_SPACE_DIM, T>(const gridPoint<PHASE_SPACE_DIM, T>& Bnd_inf, const gridPoint<PHASE_SPACE_DIM, T>& Bnd_sup, const INT& Nodes_per_dim){

		Nodes_per_Dim	= Nodes_per_dim;
		Boundary_inf	= Bnd_inf;
		Boundary_sup	= Bnd_sup;
	}

	/// @brief Create a grid specifying all the parameters (discr. length instead of the nodes per dimension)
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @param Discretization_length 
	/// @return 
	__host__ __device__	grid<PHASE_SPACE_DIM, T>(const gridPoint<PHASE_SPACE_DIM, T>& Bnd_inf, const gridPoint<PHASE_SPACE_DIM, T>& Bnd_sup, const T& Discretization_length) {

		Boundary_inf = Bnd_inf;
		Boundary_sup = Bnd_sup;
		Nodes_per_Dim = (Boundary_sup.dim[0] - Boundary_inf.dim[0]) / Discretization_length + 1;
	}

// Methods/functions
public:
	/// @brief Compute the total number of nodes
	__host__ __device__	inline UINT Total_Nodes() const {
		return pow(Nodes_per_Dim, PHASE_SPACE_DIM);
	}

	/// @brief Gives the edge length (side length of a cube)
	__host__ __device__	inline T Edge_size() const {
		return (Boundary_sup.dim[0] - Boundary_inf.dim[0]);
	}

	/// @brief Gives the center node of the grid
	__host__ __device__	inline gridPoint<PHASE_SPACE_DIM, T> Center() const {
		return (Boundary_sup + Boundary_inf).Mult_by_Scalar(0.5);
	}

	/// @brief Gives the discretization length (distance between two consecutive nodes in the same dimension) 
	__host__ __device__	inline T Discr_length() const {
		if (Nodes_per_Dim == 1) { return (T)0; }

		return (T)(this->Edge_size() / (Nodes_per_Dim - 1));
	}

	/// @brief Gives the node (point in space) given the global index in the grid
	/// @param globalIdx Global index in the current grid
	/// @return point in space
	__host__ __device__	inline gridPoint<PHASE_SPACE_DIM, T> Get_node(const INT& globalIdx) const {

		gridPoint<PHASE_SPACE_DIM, T> out(Boundary_inf);

		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			INT j = floorf( positive_rem(globalIdx, pow(Nodes_per_Dim, d + 1)) / pow(Nodes_per_Dim, d) );	// This line gives the index at each dimension

			out.dim[d] += j * this->Discr_length();															// This line gives the grid node per se
		}
		return out;
	}

	/// @brief This method decides whether Particle is inside the grid or not
	/// @param Particle 
	/// @return bool. True if particle is inside grid, false otherwise
	__host__ __device__	inline bool Contains_particle(const gridPoint<PHASE_SPACE_DIM, T>& Particle) const {
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			if (Particle.dim[d] < Boundary_inf.dim[d] || Particle.dim[d] > Boundary_sup.dim[d]) { return false; }
		}
		return true;
	}

	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	__host__ __device__ inline UINT Get_binIdx(const gridPoint<PHASE_SPACE_DIM, T>& Particle) const {
		UINT bin_idx = 0;

		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			INT temp_idx = roundf((Particle.dim[d] - Boundary_inf.dim[d]) / this->Discr_length());

			bin_idx += temp_idx * pow(Nodes_per_Dim, d);
		}
		return bin_idx;
	};
	
	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	__host__ __device__ inline UINT Get_binIdx(const gridPoint<PHASE_SPACE_DIM, T>& Particle, const INT& bin_offset) const {
		UINT bin_idx = 0;
 
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			INT temp_idx = roundf((Particle.dim[d] - Boundary_inf.dim[d]) / this->Discr_length()) + bin_offset;

			bin_idx += temp_idx * pow(Nodes_per_Dim, d);
		}
		return bin_idx;
	};

	// Compute the global index at your mesh, given the global index in "other" mesh.
	__host__ inline UINT Indx_here(const UINT& indx_at_other, const grid<PHASE_SPACE_DIM, T>& other) const {
		return this->Get_binIdx(other.Get_node(indx_at_other));
	}

	/// @brief This function expands a fixed grid "Other" by a length of  "expansion_length" in each direction/dimension
	/// @param Other The base grid from which we will expand
	/// @param expansion_nodes Number of nodes we will expand in every direction
	__host__ __device__	inline void Expand_From(const grid& Other, const UINT& expansion_nodes) {
		
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			// To make sure that the points fall into the grid nodes
			Boundary_inf.dim[d] = Other.Boundary_inf.dim[d] - Other.Discr_length() * expansion_nodes;
			Boundary_sup.dim[d] = Other.Boundary_sup.dim[d] + Other.Discr_length() * expansion_nodes;
		}

		Nodes_per_Dim = Other.Nodes_per_Dim + 2 * expansion_nodes;
	}

	/// @brief This function makes you domain a square (same Lebesgue-length in every direction)
	__host__ __device__ inline void Squarify() {
		// Get the max distance between the edges and then make the box larger!
		T max_dist = Boundary_sup.dim[0] - Boundary_inf.dim[0];

		for (uint16_t d = 1; d < PHASE_SPACE_DIM; d++) {
			max_dist = fmaxf(max_dist, Boundary_sup.dim[d] - Boundary_inf.dim[d]);
		}

		// Now that we know the max dist, let's expand the edges!
		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			Boundary_sup.dim[d] = Boundary_inf.dim[d] + max_dist;
		}
	}

	/// @brief This function updates a grid-defined bounding box
	/// @param D_Particle_Locations GPU array storing the positions of the particles
	/// @returns Nothing
	inline void Update_boundingBox(const thrust::device_vector<gridPoint<PHASE_SPACE_DIM, T>>& D_Particle_Locations){
		
		UINT Threads = fmin(THREADS_P_BLK, D_Particle_Locations.size());
		UINT Blocks	 = floor((D_Particle_Locations.size() - 1) / Threads) + 1;

		// Temporary vector storing the particles' projections in each dimension 
		thrust::device_vector<T> projection(D_Particle_Locations.size(),(T)0);

		for (uint16_t d = 0; d < PHASE_SPACE_DIM; d++) {
			findProjection<PHASE_SPACE_DIM, T> << <Blocks, Threads >> > (rpc(D_Particle_Locations, 0), rpc(projection, 0), D_Particle_Locations.size(), d);

			T temp_1 = *(thrust::min_element(thrust::device, projection.begin(), projection.end())); // min element from the projection in that direction
			T temp_2 = *(thrust::max_element(thrust::device, projection.begin(), projection.end()));


			// Eliminate the need for it to be in the Problem domain
			Boundary_inf.dim[d] = temp_1 - ceil(DISC_RADIUS) * this->Discr_length();
			Boundary_sup.dim[d] = temp_2 + ceil(DISC_RADIUS) * this->Discr_length();
		}
		projection.clear();
	}
};



#endif