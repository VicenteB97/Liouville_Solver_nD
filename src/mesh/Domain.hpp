#ifndef __DOMAIN_HPP__
#define __DOMAIN_HPP__

#include "headers.hpp"
#include "cudaBase.cuh"
#include "Particle.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class definition

class cartesianMesh{
private:
	Particle __boundary_inf, __boundary_sup;
	uint32_t __nodes_per_dim;

public:
	// Parametric constructors:
	/// @brief Create a cartesianMesh knowing the nodes per dimension
	/// @param nodes_per_dim 
	/// @return 
	hostFunction deviceFunction	
		cartesianMesh(int32_t nodes_per_dim = 2);

	/// @brief Create a cartesianMesh knowing the discretization length
	/// @param discretization_length 
	/// @return 
	hostFunction deviceFunction	
		cartesianMesh(floatType discretization_length);

	/// @brief Create a cartesianMesh specifying all the parameters
	/// @param bnd_inf 
	/// @param bnd_sup 
	/// @param nodes_per_dim 
	/// @return 
	hostFunction deviceFunction	
		cartesianMesh(const Particle& bnd_inf, const Particle& bnd_sup, int32_t nodes_per_dim = 2);

	/// @brief Create a cartesianMesh specifying all the parameters (discr. length instead of the nodes per dimension)
	/// @param bnd_inf 
	/// @param bnd_sup 
	/// @param discretization_length 
	/// @return 
	hostFunction deviceFunction	
		cartesianMesh(const Particle& bnd_inf, const Particle& bnd_sup, floatType discretization_length);

// Methods/functions
	hostFunction deviceFunction
		void set_boundary_inf(const Particle& boundary_inf);

	hostFunction deviceFunction
		Particle boundary_inf() const;

	hostFunction deviceFunction
		void set_boundary_sup(const Particle& boundary_sup);

	hostFunction deviceFunction
		Particle boundary_sup() const;

	hostFunction deviceFunction
		void set_nodes_per_dimension(uint32_t nodes_per_dim = 2);

	hostFunction deviceFunction
		uint32_t nodes_per_dim() const;

	/// @brief Compute the total number of nodes
	hostFunction deviceFunction 
		intType total_nodes() const;

	/// @brief Gives the edge length (side length of a cube)
	hostFunction deviceFunction	
		floatType edge_size() const;

	/// @brief Gives the center node of the cartesianMesh
	hostFunction deviceFunction	
		Particle center() const;

	/// @brief Gives the discretization length (distance between two consecutive nodes in the same dimension) 
	hostFunction deviceFunction	
		floatType discr_length() const;

	/// @brief Gives the node (point in space) given the global index in the cartesianMesh
	/// @param globalIdx Global index in the current cartesianMesh
	/// @return point in space
	hostFunction deviceFunction	
		Particle get_node(int32_t mesh_idx) const;

	/// @brief This method decides whether Particle is inside the cartesianMesh or not
	/// @param Particle 
	/// @return bool. True if particle is inside cartesianMesh, false otherwise
	hostFunction deviceFunction	
		bool contains_particle(const Particle& particle) const;
	
	// Returns the bin (or ID of the closest node) where Particle belongs to, adding bin_offset.
	hostFunction deviceFunction
		intType get_bin_idx(const Particle& particle, int32_t bin_offset = 0) const;

	// Compute the global index at your mesh, given the global index in "other" mesh.
	hostFunction 
		intType idx_here_from_other_mesh(int32_t indx_at_other, const cartesianMesh& other) const;

	/// @brief This function expands a fixed cartesianMesh "other" by a length of  "expansion_length" in each direction/dimension
	/// @param other The base cartesianMesh from which we will expand
	/// @param expansion_nodes Number of nodes we will expand in every direction
	hostFunction deviceFunction	
		void Expand_From(const cartesianMesh& other, int32_t expansion_nodes = 0);

	/// @brief This function makes you domain a square (same Lebesgue-length in every direction)
	hostFunction deviceFunction 
		void Squarify();

	/// @brief This function updates a cartesianMesh-defined bounding box
	/// @param D_Particle_Locations GPU array storing the positions of the particles
	/// @returns Nothing
	 hostFunction 
		 void update_bounding_box(const thrust::device_vector<Particle>& D_Particle_Locations);
};

#endif