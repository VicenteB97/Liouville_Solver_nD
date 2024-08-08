#include <Integrator.cuh>

// Dynamics functions:
// The following functions are not to be modified
__device__
inline Particle VECTOR_FIELD(
	Particle X,
	double      t,
	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter,
	const uintType      mode,
	const double    extra_param[]
) {

	return {VEC_FIELD};
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__
inline floatType DIVERGENCE_FIELD(Particle X,
	double      t,
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
	double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization, 
	const double* extra_param, 
	const uintType mode,
	const cartesianMesh domain_mesh
) {

	Particle k0, k1, k2, k3, k_final, temp;		// register storing the initial particle location;
	floatType Int1, Int2, Int3, val_aux = 1;	// register storing the initial particle value;

	while (t0 < tF - 1E-6) {
		// Just in case not detaT-proportional
		time_step = fmin(time_step, tF-t0);

		// Particle flow
		k0 = VECTOR_FIELD(position, t0, parameter_realization, mode, extra_param);

		temp = k0*(time_step / 2.00f);
		k1 = VECTOR_FIELD(position + temp, t0 + time_step / 2.00f, parameter_realization, mode, extra_param);

		temp = k1*(time_step / 2.00f);
		k2 = VECTOR_FIELD(position + temp, t0 + time_step / 2.00f, parameter_realization, mode, extra_param);

		temp = k2*(time_step);
		k3 = VECTOR_FIELD(position + temp, t0 + time_step, parameter_realization, mode, extra_param);

		k1 = k1*(2.00f);
		k2 = k2*(2.00f);

		temp = k0 + k3 + k1 + k2;
		temp = position + temp*((floatType)time_step / 6); // New particle dim

		// Integration of PDF: The following line corresponds to computing the approximation via a Hermite interpolation (we know initial and final points and their velocities)
		Int1 = DIVERGENCE_FIELD(position, t0, parameter_realization, mode, extra_param);

		k_final = VECTOR_FIELD(temp, t0 + time_step, parameter_realization, mode, extra_param);
		position = (position + temp)*(0.5);
		position = position + (k0 + k_final)*(0.125);
		Int2 = DIVERGENCE_FIELD(position, (floatType)(2 * t0 + time_step) / 2, parameter_realization, mode, extra_param);

		Int3 = DIVERGENCE_FIELD(temp, (floatType)t0 + time_step, parameter_realization, mode, extra_param);

		val_aux *= expf((float)-time_step / 6 * (Int1 + 4 * Int2 + Int3)); // New particle value

		// Reinit step
		position = temp;
		t0 += time_step;
	}

	value *= val_aux;

	if (!domain_mesh.contains_particle(position)) { value = 0; }
}

/// @brief This case is ONLY VALID for 2D methods! We'll use the 
/// @param position 
/// @param value 
/// @param t0 
/// @param tF 
/// @param time_step 
/// @param parameter_realization 
/// @param extra_param 
/// @param mode 
/// @param domain_mesh 
/// @return 
__device__ void lie_euler_mathieu(
	Particle& position,
	floatType& value,
	double t0,
	const double tF,
	double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization,
	const double* extra_param,
	const uintType mode,
	const cartesianMesh domain_mesh
){

	floatType val_aux = 1;
	// The Euler step:
	while (t0 < tF - 1E-6){
		// Just in case not detaT-proportional
		double h {fmin(time_step, tF-t0)};

		Particle u_n {position}, temp_position;

		floatType c {(floatType) -(parameter_realization.sample_vec[0]-2*parameter_realization.sample_vec[1]*cos(2*t0))};

		temp_position.dim[0] = (1 + 1/24 *c * h*h *(12 + c *h*h)) * u_n.dim[0] + (h + (c * h*h*h)/6 + (c*c*powf(h,5))/120) * u_n.dim[1];
		temp_position.dim[1] = (c * h + 1/120*c*c*h*h*h* (20 + c * h*h)) * u_n.dim[0] + (1 + 1/24 * c * h*h * (12 + c * h*h)) * u_n.dim[1];


		// Advect the integral value:
		floatType Int1 {DIVERGENCE_FIELD(position, t0, parameter_realization, mode, extra_param)};

		Particle k0 {VECTOR_FIELD(position, t0, parameter_realization, mode, extra_param)};
		Particle k_final {VECTOR_FIELD(temp_position, t0 + time_step, parameter_realization, mode, extra_param)};

		position = (position + temp_position)*(0.5);
		position = position + (k0 + k_final)*(0.125);

		floatType Int2 {DIVERGENCE_FIELD(position, (floatType)(2 * t0 + time_step) / 2, parameter_realization, mode, extra_param)};

		floatType Int3 {DIVERGENCE_FIELD(temp_position, (floatType)t0 + time_step, parameter_realization, mode, extra_param)};

		val_aux *= expf((float)-time_step / 6 * (Int1 + 4 * Int2 + Int3)); // New particle value


		// Reinit step
		position = temp_position;
		t0 += h;
	}

	value *= val_aux;

	if (!domain_mesh.contains_particle(position)) { value = 0; }
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
	const parameterPair* parameters,
	const intType* n_Samples,
	double t0,
	const double time_step,
	const double tF,
	const intType Adapt_Points,
	const intType Random_Samples,
	const uintType mode,
	const double* extra_param,
	const cartesianMesh D_cartesianMesh) {

	const uint64_t i {blockDim.x * blockIdx.x + threadIdx.x};

	if (i < Adapt_Points * Random_Samples) {

		// AUXILIARY DATA TO RUN THE ITERATIONS
		// So, the total amount of advections are going to be: (no. particles x no. of samples)
		const uintType  i_sample {(uintType) floor((double)i / Adapt_Points)};

		const Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization {Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i_sample, parameters, n_Samples)};

		Particle pos_particle {Particles[i]};
		floatType value_particle {PDF[i]};

		#if SPECIAL_INTEGRATOR
		lie_euler_mathieu(pos_particle, value_particle, t0, tF, time_step, parameter_realization, extra_param, mode, D_cartesianMesh);
		#else
		runge_kutta_45(pos_particle, value_particle, t0, tF, time_step, parameter_realization, extra_param, mode, D_cartesianMesh);
		#endif

		// Output results
		Particles[i] = pos_particle;
		PDF[i] = value_particle;
	}
}
