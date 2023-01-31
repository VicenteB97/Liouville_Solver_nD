#ifndef __DYNAMICS_CUH__
#define __DYNAMICS_CUH__


#include "Classes.cuh"

//-------------------------------------------------------------------------
__device__ gridPoint VECTOR_FIELD(gridPoint X, double t, const Param_vec parameter) {
	gridPoint output;

	// write the corresponding vector field
	// DESCRIPTION: UNFORCED DUFFING OSCILLATOR
	// parameter[0] = Xi
	// parameter[1] = Lambda
	// parameter[2] = Forcing

	output.position[0] = X.position[1];
	output.position[1] = -2 * parameter.sample_vec[0] * X.position[1] - X.position[0] - parameter.sample_vec[1] * pow(X.position[0], 3); // + parameter.sample_vec[2]; for adding external forcing

	return output;
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__ double DIVERGENCE_FIELD(gridPoint X, double t, const Param_vec parameter) {
	double output;

	// write the value of the divergence of the vector field
	// DESCRIPTION: UNFORCED DUFFING OSCILLATOR
	// parameter[0] = Xi
	// parameter[1] = Lambda

	output = -2*parameter.sample_vec[0];

	return output;
}
//-------------------------------------------------------------------------
#endif