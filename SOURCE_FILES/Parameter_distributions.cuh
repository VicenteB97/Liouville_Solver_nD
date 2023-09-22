//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
// This part contains the definitions of the functions that will be used to build the parameter mesh + j-PDF   //
// We STRONGLY RECOMMEND not changing absolutely anything if your only use of this program is to simulate	   //
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//

#ifndef __PARAMETER_DISTRIBUTIONS_CUH__
#define __PARAMETER_DISTRIBUTIONS_CUH__

#include "Classes.cuh"

#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/beta.hpp>
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// --------------------- TO BUILD THE INITIAL CONDITION! ----------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //

/// @brief This function creates the IC PDF. We assume it is the tensor product of Normal Distributions
/// @param Points_per_dimension 
/// @param Mesh 
/// @param PDF_value 
/// @param IC_dist_parameters 
int16_t PDF_INITIAL_CONDITION(UINT Points_per_dimension, const grid& Mesh, thrust::host_vector<TYPE>& PDF_value, TYPE* IC_dist_parameters) {

	// Due to obvious reasons, we will not make the choice of IC distributions automatically, as with the model parameters
	boost::math::normal dist[DIMENSIONS];

	//create the distributions per dimension:
	 
	for (UINT d = 0; d < DIMENSIONS; d++){
		dist[d] = boost::math::normal_distribution((FIXED_TYPE)IC_dist_parameters[2*d], (FIXED_TYPE)IC_dist_parameters[2*d + 1]);
	}

#pragma omp parallel for
	for (INT k = 0; k < PDF_value.size(); k++){
		TYPE aux = 1;

		for (UINT d = 0; d < DIMENSIONS; d++){
			aux *= boost::math::pdf(dist[d], Mesh.Get_node(k).dim[d]);
		}
		PDF_value[k] = aux; // with positive and negative parts!
	}

	return 0;
}


// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// ------------------ TO BUILD THE RANDOM PARAMETER MESH! ---------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //

/// @brief 
/// @param n_Samples 
/// @param PP 
/// @param Dist_params 
int16_t PARAMETER_VEC_BUILD(const int n_Samples, Param_pair* PP, const Distributions Dist_params) {

	TYPE expectation = Dist_params.params[0], std_dev = Dist_params.params[1];

	if (n_Samples == 1) {
		*PP = { expectation, 1 };
		return 0;
	}

	TYPE x0, xF, dx;
	
	// Normal case
	if (Dist_params.Name == 'N' || Dist_params.Name == 'n'){
		// Build the distribution:
		auto dist = boost::math::normal_distribution(expectation, std_dev);

		x0 = expectation - 6 * std_dev;
		xF = expectation + 6 * std_dev;

		
		if(Dist_params.Truncated){
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			TYPE rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// Mesh discretization
			dx = (xF - x0) / (n_Samples - 1);

			for (int i = 0; i < n_Samples; i++) {
				TYPE x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (n_Samples - 1);

		for (int i = 0; i < n_Samples; i++) {
			TYPE x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	// Uniform case
	if(Dist_params.Name == 'U' || Dist_params.Name == 'u'){

		x0 = expectation - sqrtf(3) * std_dev;
		xF = expectation + sqrtf(3) * std_dev;

		auto dist = boost::math::uniform_distribution(x0, xF);

		dx = (xF - x0) / (n_Samples - 1);
		for (int i = 0; i < n_Samples; i++) {
			TYPE x = x0 + i * dx;
			PP[i] = {x , boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	// Gamma case
	if(Dist_params.Name == 'G' || Dist_params.Name == 'g'){

		TYPE shape = pow(expectation / std_dev, 2);
		TYPE scale = pow(std_dev, 2) / expectation;

		auto dist = boost::math::gamma_distribution(shape, scale);

		x0 = fmaxf(0, expectation - 6 * std_dev);
		xF = expectation + 6 * std_dev;

		
		if(Dist_params.Truncated){
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			TYPE rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// Mesh discretization
			dx = (xF - x0) / (n_Samples - 1);

			for (int i = 0; i < n_Samples; i++) {
				TYPE x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (n_Samples - 1);

		for (int i = 0; i < n_Samples; i++) {
			TYPE x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}
	
	// Beta case
	if(Dist_params.Name == 'B' || Dist_params.Name == 'b'){

		TYPE shape_1 = -expectation * (1-(1 - expectation)/std_dev*expectation/std_dev);
		TYPE shape_2 = (expectation - 1) * (1-(1 - expectation)/std_dev*expectation/std_dev);

		auto dist = boost::math::beta_distribution(shape_1, shape_2);

		x0 = fmaxf(0, expectation - 6 * std_dev);
		xF = fminf(1, expectation + 6 * std_dev);

		
		if(Dist_params.Truncated){
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			TYPE rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// Mesh discretization
			dx = (xF - x0) / (n_Samples - 1);

			for (int i = 0; i < n_Samples; i++) {
				TYPE x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (n_Samples - 1);

		for (int i = 0; i < n_Samples; i++) {
			TYPE x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	std::cout << "Parameter distribution not recognized. Choose correctly please. \nExiting program...\n";
	return -1;
}

/// @brief This function builds the Parameter Mesh that will be used in the Liouville Solver 
/// @param n_samples Array where the number of samples per parameter is stored 
/// @param Parameter_Mesh Parameter Mesh 
/// @param Dist_Parameters Parameters' (hyper)parameters
/// @param Dist_Names Distributions that will be assigned (N = Normal, U = Uniform, etc.)
int16_t RANDOMIZE(const INT* 		n_samples, 
				Param_pair* 			Parameter_Mesh, 
				const Distributions* 	Dist_Parameters) {

	UINT aux = 0;

	 
	for (UINT d = 0; d < PARAM_DIMENSIONS; d++){
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];

		int16_t err_check = PARAMETER_VEC_BUILD(n_samples[d], PP, Dist_Parameters[d]);
		if (err_check == -1){ return -1; }

		std::copy_n(&PP[0], n_samples[d], &Parameter_Mesh[aux]);

		delete[] PP;
		aux += n_samples[d];
	}

	return 0;
}

#endif