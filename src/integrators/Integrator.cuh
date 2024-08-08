#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

#include "headers.cuh"
#include "Probability.cuh"
#include "Domain.cuh"
#include "utils/numeric_defs.cuh"

// Dynamics functions:
// The following functions are not to be modified
__device__
inline Particle VECTOR_FIELD(
	Particle X,
	double      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType      mode,
	const double    extra_param[]);
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__
inline floatType DIVERGENCE_FIELD(
	Particle X,
	double      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType   mode,
	const double extra_param[]);


// Output the final point
__device__ void runge_kutta_45(Particle& position, floatType& value, double t0, const double tF, const double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization, const double* extra_param, const uintType mode,
	const cartesianMesh domain_mesh);

__device__ void lie_midpoint_mathieu(
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
__global__ void ODE_INTEGRATE(
	Particle* Particles,
	floatType* PDF,
	const parameterPair* parameters,
	const intType* n_Samples,
	double			t0,
	const double	time_step,
	const double	tF,
	const intType		Adapt_Points,
	const intType		Random_Samples,
	const uintType		mode,
	const double* extra_param,
	const cartesianMesh  	D_cartesianMesh);

#endif