#ifndef __DYNAMICS_CUH__
#define __DYNAMICS_CUH__


#include "Classes.cuh"

//-------------------------------------------------------------------------
__device__ inline gridPoint VECTOR_FIELD(gridPoint X, double t, const Param_vec parameter) {
	gridPoint output;

	#if (CASE == 1)
		output.dim[0] = parameter.sample_vec[0] * X.dim[0] + parameter.sample_vec[1];
	#endif
	#if(CASE == 2)
		output.dim[0] = X.dim[1];
		output.dim[1] = -2 * parameter.sample_vec[0] * X.dim[1] - X.dim[0] - parameter.sample_vec[1] * pow(X.dim[0], 3); // + parameter.sample_vec[2]; for adding external forcing
	#endif
	
	return output;
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
__device__ inline double DIVERGENCE_FIELD(gridPoint X, double t, const Param_vec parameter) {
	double output;

	#if (CASE == 1)
		output = parameter.sample_vec[0];
	#endif
	#if (CASE == 2)
		output = -2*parameter.sample_vec[0];
	#endif
	
	return output;
}
//-------------------------------------------------------------------------
#endif