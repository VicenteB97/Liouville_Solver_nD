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
__device__ void forward_runge_kutta_45(
	Particle& position, 
	floatType& value, 
	double t0, 
	const double tF, 
	double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization, 
	const double* extra_param, 
	const uintType mode,
	const Mesh domain_mesh
) {

	Particle k0, k1, k2, k3, k_final, temp;		// register storing the initial particle location;
	floatType Int1, Int2, Int3, val_aux = 1;	// register storing the initial particle value;
	double t = t0;

	while (t < tF - 1E-6) {
		// Just in case not detaT-proportional
		time_step = fmin(time_step, tF-t);

		// Particle flow
		k0 = VECTOR_FIELD(position, t, parameter_realization, mode, extra_param);

		temp = k0.Mult_by_Scalar(time_step / 2.00f);
		k1 = VECTOR_FIELD(position + temp, t + time_step / 2.00f, parameter_realization, mode, extra_param);

		temp = k1.Mult_by_Scalar(time_step / 2.00f);
		k2 = VECTOR_FIELD(position + temp, t + time_step / 2.00f, parameter_realization, mode, extra_param);

		temp = k2.Mult_by_Scalar(time_step);
		k3 = VECTOR_FIELD(position + temp, t + time_step, parameter_realization, mode, extra_param);

		k1 = k1.Mult_by_Scalar(2.00f);
		k2 = k2.Mult_by_Scalar(2.00f);

		temp = k0 + k3 + k1 + k2;
		temp = position + temp.Mult_by_Scalar((floatType)time_step / 6); 	// New particle dims

		// Reinit step
		position = temp;
		t += time_step;
	}
}

// Output the final point
__device__ void inverse_runge_kutta_45(
	Particle& position, 
	floatType& value, 
	double t0, 
	const double tF, 
	double time_step,
	Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization, 
	const double* extra_param, 
	const uintType mode,
	const Mesh domain_mesh
) {

	Particle k0, k1, k2, k3, k_final, temp;		// register storing the initial particle location;
	floatType Int1, Int2, Int3, val_aux = 1;	// register storing the initial particle value;
	double t = t0;

	while (t < tF - 1E-6) {
		// Just in case not detaT-proportional
		time_step = fmin(time_step, tF - t);

		// Particle flow
		k0 = VECTOR_FIELD(position, tF + t0 - t, parameter_realization, mode, extra_param);
		k0 = k0.Mult_by_Scalar(-1.0);

		temp = k0.Mult_by_Scalar(time_step / 2.00f);
		k1 = -VECTOR_FIELD(position + temp, tF + t0 - t + time_step / 2.00f, parameter_realization, mode, extra_param);
		k1 = k1.Mult_by_Scalar(-1.0);

		temp = k1.Mult_by_Scalar(time_step / 2.00f);
		k2 = -VECTOR_FIELD(position + temp, tF + t0 - t + time_step / 2.00f, parameter_realization, mode, extra_param);
		k2 = k2.Mult_by_Scalar(-1.0);

		temp = k2.Mult_by_Scalar(time_step);
		k3 = -VECTOR_FIELD(position + temp, tF + t0 - t + time_step, parameter_realization, mode, extra_param);
		k3 = k3.Mult_by_Scalar(-1.0);

		k1 = k1.Mult_by_Scalar(2.00f);
		k2 = k2.Mult_by_Scalar(2.00f);

		temp = k0 + k3 + k1 + k2;
		temp = position + temp.Mult_by_Scalar((floatType)time_step / 6); 	// New particle dims

		// Reinit step
		position = temp;
		t += time_step;
	}
}

// NEW FUNCTIONS: (Forward + backward position integration) + (Backward position integration (and get value from the wavelet transform?)) 
__global__ void forward_integrate_positions(
	Particle* Particles,
	const Param_pair* parameters,
	const intType* n_Samples,
	double t0,
	const double time_step,
	const double tF,
	const intType Adapt_Points,
	const intType Random_Samples,
	const uintType mode,
	const double* extra_param,
	const Mesh D_Mesh) {

	const uint64_t i {blockDim.x * blockIdx.x + threadIdx.x};

	if (i >= Adapt_Points * Random_Samples) { return; }
	// AUXILIARY DATA TO RUN THE ITERATIONS
	// So, the total amount of advections are going to be: (no. particles x no. of samples)
	const uintType  i_sample {(uintType) floor((double)i / Adapt_Points)};

	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization {
		Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i_sample, parameters, n_Samples)
	};

	Particle pos_particle {Particles[i]};
	forward_runge_kutta_45(
		pos_particle, t0, tF, time_step, parameter_realization, extra_param, mode, D_Mesh
	);

	// Output results
	Particles[i] = pos_particle;
}


__global__ void inverse_integrate_system(
	Particle* Particles,
	floatType* Values,
	const Param_pair* parameters,
	const intType* n_Samples,
	double t0,
	const double time_step,
	const double tF,
	const intType Adapt_Points,
	const intType Random_Samples,
	const uintType mode,
	const double* extra_param,
	const Mesh D_Mesh) {

	const uint64_t i {blockDim.x * blockIdx.x + threadIdx.x};

	if (i >= Adapt_Points * Random_Samples) { return; }
	// AUXILIARY DATA TO RUN THE ITERATIONS
	// So, the total amount of advections are going to be: (no. particles x no. of samples)
	const uintType  i_sample {(uintType) floor((double)i / Adapt_Points)};

	const Param_vec<PARAM_SPACE_DIMENSIONS> parameter_realization {
		Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i_sample, parameters, n_Samples)
	};

	Particle pos_particle {Particles[i]};	
	inverse_runge_kutta_45(
		pos_particle, t0, tF, time_step, parameter_realization, extra_param, mode, D_Mesh
	);

	// Output results
	Particles[i] = pos_particle;
}
