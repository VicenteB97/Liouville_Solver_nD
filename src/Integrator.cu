#include <./include/Integrator.cuh>

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

// Dynamics functions:
// The following functions are not to be modified
__device__
inline Particle VECTOR_FIELD(
	Particle X,
	floatType      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType      mode,
	const double    extra_param[]
) {

	return { VEC_FIELD };
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__
inline floatType DIVERGENCE_FIELD(Particle X,
	floatType      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType   mode,
	const double extra_param[]) {

	return DIVERGENCE;
}


// Output the final point
__device__ void runge_kutta_45(
	Particle& position, 
	floatType& value, 
	double t0, 
	const double tF, 
	const double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization, 
	const double* extra_param, 
	const uintType mode,
	const Mesh domain_mesh
) {

	Particle k0, k1, k2, k3, k_final, temp;	// register storing the initial particle location;
	floatType	 Int1, Int2, Int3;				// register storing the initial particle value;

	while (t0 < tF - time_step / 2) {
		// Particle flow
		k0 = VECTOR_FIELD(position, t0, parameter_realization, mode, extra_param);

		temp = k0.Mult_by_Scalar(time_step / 2.00f);
		k1 = VECTOR_FIELD(position + temp, t0 + time_step / 2.00f, parameter_realization, mode, extra_param);

		temp = k1.Mult_by_Scalar(time_step / 2.00f);
		k2 = VECTOR_FIELD(position + temp, t0 + time_step / 2.00f, parameter_realization, mode, extra_param);

		temp = k2.Mult_by_Scalar(time_step);
		k3 = VECTOR_FIELD(position + temp, t0 + time_step, parameter_realization, mode, extra_param);

		k1 = k1.Mult_by_Scalar(2.00f);
		k2 = k2.Mult_by_Scalar(2.00f);

		temp = k0 + k3 + k1 + k2;
		temp = position + temp.Mult_by_Scalar((floatType)time_step / 6.00f); // New particle dim

		// Integration of PDF: The following line corresponds to computing the approximation via a Hermite interpolation (we know initial and final points and their velocities)
		Int1 = DIVERGENCE_FIELD(position, t0, parameter_realization, mode, extra_param);

		k_final = VECTOR_FIELD(temp, t0 + time_step, parameter_realization, mode, extra_param);
		position = (position + temp).Mult_by_Scalar(0.50f);
		position = position + (k0 + k_final).Mult_by_Scalar(0.125f);
		Int2 = DIVERGENCE_FIELD(position, (floatType)(2.00f * t0 + time_step) / 2.00f, parameter_realization, mode, extra_param);

		Int3 = DIVERGENCE_FIELD(temp, (floatType)t0 + time_step, parameter_realization, mode, extra_param);

		value *= expf((float)-time_step / 6.00f * (Int1 + 4.00f * Int2 + Int3)); // New particle value

		// Reinit step
		position = temp;
		t0 += time_step;
	}

	if (!domain_mesh.Contains_particle(position)) { value = 0; }
}


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
	const Param_pair* parameters,
	const intType* n_Samples,
	double			t0,
	const double	time_step,
	const double	tF,
	const intType		Adapt_Points,
	const intType		Random_Samples,
	const uintType		mode,
	const double* extra_param,
	const Mesh  	D_Mesh) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Adapt_Points * Random_Samples) {

		// AUXILIARY DATA TO RUN THE ITERATIONS
		// So, the total amount of advections are going to be: (no. particles x no. of samples)
		const uintType  i_sample = (uintType) floorf((float)i / Adapt_Points);

		const Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i_sample, parameters, n_Samples);

		Particle pos_particle = Particles[i];
		floatType 	 value_particle = PDF[i];

		runge_kutta_45(pos_particle, value_particle, t0, tF, time_step, parameter_realization, extra_param, mode, D_Mesh);

		// Output results
		Particles[i] = pos_particle;
		PDF[i] = value_particle;
	}
}
