#include "Particle.cuh"

// Default constructor
hostFunction deviceFunction
Particle::Particle() {
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		dim[d] = (floatType)0;
	}
}

// Parametric constructors
hostFunction deviceFunction
Particle::Particle(const floatType(&input)[PHASE_SPACE_DIMENSIONS]) {
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		dim[d] = input[d];
	}
}

hostFunction deviceFunction
Particle::Particle(const Particle& input) {
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		dim[d] = input.dim[d];
	}
}

// Operators and methods
hostFunction deviceFunction
Particle Particle::operator+(const Particle& other) const {

	Particle out;

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		out.dim[d] = dim[d] + other.dim[d];
	}

	return out;
}
hostFunction deviceFunction
Particle Particle::operator-(const Particle& other) const {
	Particle out;

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		out.dim[d] = dim[d] - other.dim[d];
	}

	return out;
}

hostFunction deviceFunction
void Particle::operator=(const Particle& other) {
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		dim[d] = other.dim[d];
	}
}

hostFunction deviceFunction
bool Particle::operator==(const Particle& other) const {
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		if (dim[d] != other.dim[d]) { return false; }
	}
	return true;
}

hostFunction deviceFunction
floatType Particle::distance(const Particle& other) const {
	floatType dist = 0;
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		dist += (dim[d] - other.dim[d]) * (dim[d] - other.dim[d]);
	}
	return sqrtf(dist);
}


hostFunction deviceFunction
Particle Particle::mult_by_scalar(floatType scalar) const {
	Particle out;

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		out.dim[d] = scalar * dim[d];
	}

	return out;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid class ----------------------------------------------------------------------------------------------------------------- //

// This function is defined aside because CUDA does not allow defining __global__ functions inside class definitions! (At least not statically)

void find_projection::operator()(const uint64_t global_id) {
	if (global_id >= in_total_particles) { return; }
	out_projections[global_id] = in_particles[global_id].dim[in_project_dimension];
}