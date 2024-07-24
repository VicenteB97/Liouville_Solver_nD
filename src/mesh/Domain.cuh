#ifndef __DOMAIN_CUH__
#define __DOMAIN_CUH__

#include "headers.cuh"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid points ---------------------------------------------------------------------------------------------------------------- //

class Particle {
public:
	floatType dim[PHASE_SPACE_DIMENSIONS];

public:
	// Default constructor
	__host__ __device__
		Particle();

	// Parametric constructors
	__host__ __device__
		Particle(const floatType(&input)[PHASE_SPACE_DIMENSIONS]);

	__host__ __device__
		Particle(const Particle& input);

	// Operators and methods
	__host__ __device__
		Particle operator+(const Particle& other) const;

	__host__ __device__
		Particle operator-(const Particle& other) const;

	__host__ __device__
		bool operator==(const Particle& other) const;

	__host__ __device__ 
		void operator=(const Particle &other);

	__host__ __device__
		floatType Distance(const Particle& other) const;
	
	__host__ __device__
		Particle Mult_by_Scalar(floatType scalar) const;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class ----------------------------------------------------------------------------------------------------------------- //

// This function is defined aside because CUDA does not allow defining __global__ functions inside class definitions! (At least not statically)
 
__global__ void findProjection(const Particle* particles, floatType* projections, intType totalParticles, intType dimension);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class definition

class Mesh{
public:
	Particle 	Boundary_inf, Boundary_sup;
	intType			Nodes_per_Dim;

	// Parametric constructors:
	/// @brief Create a Mesh knowing the nodes per dimension
	/// @param Nodes_per_dim 
	/// @return 
	__host__ __device__	Mesh(intType Nodes_per_dim = 2);

	/// @brief Create a Mesh knowing the discretization length
	/// @param Discretization_length 
	/// @return 
	__host__ __device__	Mesh(floatType Discretization_length);

	/// @brief Create a Mesh specifying all the parameters
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @param Nodes_per_dim 
	/// @return 
	__host__ __device__	Mesh(const Particle& Bnd_inf, const Particle& Bnd_sup, intType Nodes_per_dim = 2);

	/// @brief Create a Mesh specifying all the parameters (discr. length instead of the nodes per dimension)
	/// @param Bnd_inf 
	/// @param Bnd_sup 
	/// @param Discretization_length 
	/// @return 
	__host__ __device__	Mesh(const Particle& Bnd_inf, const Particle& Bnd_sup, floatType Discretization_length);

// Methods/functions
public:
	/// @brief Compute the total number of nodes
	__host__ __device__	 intType Total_Nodes() const;

	/// @brief Gives the edge length (side length of a cube)
	__host__ __device__	 floatType Edge_size() const;

	/// @brief Gives the center node of the Mesh
	__host__ __device__	 Particle Center() const;

	/// @brief Gives the discretization length (distance between two consecutive nodes in the same dimension) 
	__host__ __device__	 floatType Discr_length() const;

	/// @brief Gives the node (point in space) given the global index in the Mesh
	/// @param globalIdx Global index in the current Mesh
	/// @return point in space
	__host__ __device__	 Particle Get_node(intType globalIdx) const;

	/// @brief This method decides whether Particle is inside the Mesh or not
	/// @param Particle 
	/// @return bool. True if particle is inside Mesh, false otherwise
	__host__ __device__	 bool Contains_particle(const Particle& Particle) const;
	
	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	__host__ __device__  intType Get_binIdx(const Particle& Particle, intType bin_offset = 0) const;

	// Compute the global index at your mesh, given the global index in "other" mesh.
	__host__  intType Indx_here(intType indx_at_other, const Mesh& other) const;

	/// @brief This function expands a fixed Mesh "Other" by a length of  "expansion_length" in each direction/dimension
	/// @param Other The base Mesh from which we will expand
	/// @param expansion_nodes Number of nodes we will expand in every direction
	__host__ __device__	 void Expand_From(const Mesh& Other, intType expansion_nodes);

	/// @brief This function makes you domain a square (same Lebesgue-length in every direction)
	__host__ __device__  void Squarify();

	/// @brief This function updates a Mesh-defined bounding box
	/// @param D_Particle_Locations GPU array storing the positions of the particles
	/// @returns Nothing
	 __host__ void Update_boundingBox(const thrust::device_vector<Particle>& D_Particle_Locations);
};

#endif
