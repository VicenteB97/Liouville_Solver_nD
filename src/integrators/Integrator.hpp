#ifndef __INTEGRATOR_HPP__
#define __INTEGRATOR_HPP__

#include "include/headers.hpp"
#include "probabilityDistributions/Probability.hpp"
#include "mesh/Domain.hpp"
#include "mesh/Particle.hpp"

// Dynamics functions:
// The following functions are not to be modified
deviceFunction
inline Particle VECTOR_FIELD(
	Particle X,
	double      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType      mode,
	const double    extra_param[]);
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
deviceFunction
inline floatType DIVERGENCE_FIELD(
	Particle X,
	double      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType   mode,
	const double extra_param[]);


// Output the final point
deviceFunction void runge_kutta_45(
	Particle& position,
	floatType& value,
	double t0,
	const double tF,
	double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization,
	const double* extra_param,
	const uintType mode,
	const cartesianMesh domain_mesh
);

deviceFunction void lie_midpoint_mathieu(
	Particle& position,
	floatType& value,
	double t0,
	const double tF,
	double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization,
	const double* extra_param,
	const uintType mode,
	const cartesianMesh domain_mesh
);


/// @brief This function computes the advection of the particles created by AMR.
/// @param Particles Particle location (spatial variables)
/// @param PDF PDF value at the corresponding particle location
/// @param parameters Parameters to be used for the vector field and its corresponding divergence function
/// @param t0 Inital time for starting the simulation
/// @param time_step Time step used in the simulation
/// @param ReinitSteps Number of steps before needing a re-interpolation
/// @param Adapt_Points Number of particles as computed by the AMR scheme
/// @param Random_Samples Number of random parameter_realization samples
/// @return 


class characteristicIntegrator {
public:
	Particle* particleLocations;
	floatType* particleValues;
	const parameterPair* modelParameters;
	const intType* sampleCountPerParameter;
	double t0;
	const double time_step;
	const double tF;
	const uint32_t particleCountPerSample;
	const uint32_t sampleCountTotal;
	const uint16_t fieldModeIndex;
	const double* extraModelParameters;
	const cartesianMesh problemDomain;
public:
	deviceFunction void operator()(const uint64_t global_id) const;
};

#endif