#ifndef __PARTICLE_HPP__
#define __PARTICLE_HPP__

#include "include/headers.hpp"

class Particle {
public:
	floatType dim[PHASE_SPACE_DIMENSIONS];

public:
	// Default constructor
	hostFunction deviceFunction
		Particle();

	// Parametric constructors
	hostFunction deviceFunction
		Particle(const floatType(&input)[PHASE_SPACE_DIMENSIONS]);

	hostFunction deviceFunction
		Particle(const Particle& input);

	// Operators and methods
	hostFunction deviceFunction
		Particle operator+(const Particle& other) const;

	hostFunction deviceFunction
		Particle operator-(const Particle& other) const;

	hostFunction deviceFunction
		bool operator==(const Particle& other) const;

	hostFunction deviceFunction
		void operator=(const Particle& other);

	hostFunction deviceFunction
		Particle operator*(floatType scalar);

	hostFunction deviceFunction
		floatType distance(const Particle& other) const;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class ----------------------------------------------------------------------------------------------------------------- //

// This function is defined aside because CUDA does not allow defining __global__ functions inside class definitions! (At least not statically)
class find_projection {
public:
	const Particle* in_particles;
	floatType* out_projections;
	uint64_t in_total_particles;
	uint64_t in_project_dimension;

public:
	void operator()(const uint64_t global_id);
};

#endif