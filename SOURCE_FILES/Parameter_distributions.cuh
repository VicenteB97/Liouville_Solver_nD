#ifndef __PARAMETER_DISTRIBUTIONS_CUH__
#define __PARAMETER_DISTRIBUTIONS_CUH__

#include "Classes.cuh"

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
void PDF_INITIAL_CONDITION(int Points_per_dimension, const gridPoint* Mesh, thrust::host_vector<double>& PDF_value, double* IC_dist_parameters) {

	boost::math::normal dist[DIMENSIONS];

	//create the distributions per dimension:
	#pragma unroll
	for (unsigned int d = 0; d < DIMENSIONS; d++){
		dist[d] = boost::math::normal_distribution(IC_dist_parameters[2*d], IC_dist_parameters[2*d + 1]);
	}

	for (unsigned int k = 0; k < PDF_value.size(); k++){
		double aux = 1;
		for (unsigned d = 0; d < DIMENSIONS; d++){
			aux *= boost::math::pdf(dist[d], Mesh[k].dim[d]);
		}
		PDF_value[k] = aux;
	}
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
int PARAMETER_VEC_BUILD(const int n_Samples, Param_pair* PP, const Distributions Dist_params) {

	double expectation = Dist_params.params[0], std_dev = Dist_params.params[1];

	if (n_Samples == 1) {
		*PP = { expectation, 1 };
		return 0;
	}
	else {

		if (Dist_params.Name == 'N'){
			// Build the distribution:
			auto dist = boost::math::normal_distribution(expectation, std_dev);

			if (Dist_params.Truncated){
				// Make the truncation intervals:
				double x0 = fmaxf(expectation - 6 * std_dev, Dist_params.trunc_interval[0]);
				double xF = fminf(expectation + 6 * std_dev, Dist_params.trunc_interval[1]);

				// Re-scaling for the truncation of the random variables
				double rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

				// Mesh discretization
				double dx = (xF - x0) / (n_Samples - 1);

				for (int i = 0; i < n_Samples; i++) {
					double x = x0 + i * dx;
					PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
				}
			}
			else{
				
				double x0 = expectation - 6 * std_dev;
				double xF = expectation + 6 * std_dev;

				double dx = (xF - x0) / (n_Samples - 1);

				for (int i = 0; i < n_Samples; i++) {
					double x = x0 + i * dx;
					PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
				}
			}
			return 0;
		}
		else if(Dist_params.Name == 'U'){

			double x0 = expectation - sqrtf(3) * std_dev;
			double xF = expectation + sqrtf(3) * std_dev;

			auto dist = boost::math::uniform_distribution(x0, xF);

			double dx = (xF - x0) / (n_Samples - 1);
			for (int i = 0; i < n_Samples; i++) {
				double x = x0 + i * dx;
				PP[i] = {x , boost::math::pdf(dist, x) }; // other distributions could be used
			}
			return 0;
		}
		else if(Dist_params.Name == 'G'){

			double shape = pow(expectation / std_dev, 2);
			double scale = pow(std_dev, 2) / expectation;

			auto dist = boost::math::gamma_distribution(shape, scale);

			if(Dist_params.Truncated){
				double x0 = fmax(fmax(0, expectation - 6 * std_dev), Dist_params.trunc_interval[0]);
				double xF = fmin(expectation + 6 * std_dev, Dist_params.trunc_interval[1]);

				// Re-scaling for the truncation of the random variables
				double rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

				// Mesh discretization
				double dx = (xF - x0) / (n_Samples - 1);

				for (int i = 0; i < n_Samples; i++) {
					double x = x0 + i * dx;
					PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
				}
			}
			else{
				double x0 = fmax(0, expectation - 6 * std_dev);
				double xF = expectation + 6 * std_dev;

				double dx = (xF - x0) / (n_Samples - 1);

				for (int i = 0; i < n_Samples; i++) {
					double x = x0 + i * dx;
					PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
				}
			}
			return 0;
		}
		else{
			std::cout << "Parameter distribution not recognized. Choose correctly please. \nExiting program...\n";
			return -1;
		}
	}
}

/// @brief This function builds the Parameter Mesh that will be used in the Liouville Solver 
/// @param n_samples Array where the number of samples per parameter is stored 
/// @param Total_Samples Total number of samples
/// @param Parameter_Mesh Parameter Mesh 
/// @param Dist_Parameters Parameters' (hyper)parameters
/// @param Dist_Names Distributions that will be assigned (N = Normal, U = Uniform, etc.)
int RANDOMIZE(	const int* 				n_samples, 
				const int 				Total_Samples, 
				std::vector<Param_vec>* Parameter_Mesh, 
				const Distributions* 	Dist_Parameters) {

	std::vector<Param_pair> aux_PM;

	#pragma unroll
	for (unsigned int d = 0; d < PARAM_DIMENSIONS; d++){
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];

		int err_check = PARAMETER_VEC_BUILD(n_samples[d], PP, Dist_Parameters[d]);
		if (err_check == -1){ return -1; }

		// append to the output array
		aux_PM.insert(aux_PM.end(), &PP[0], &PP[n_samples[d]]);
		delete[] PP;
	}

	for (unsigned int k = 0; k < Total_Samples; k++){
		// 1st, find the parameter components
		int aux_num 	=  n_samples[0];
		int aux_num_2 	=  n_samples[0];

		int aux_idx = positive_rem(k, aux_num);

		Parameter_Mesh->at(k).sample_vec[0] = aux_PM[aux_idx].sample;
		Parameter_Mesh->at(k).Joint_PDF 	= aux_PM[aux_idx].PDF;

		#pragma unroll
		for (unsigned int d = 1; d < PARAM_DIMENSIONS; d++){

			aux_idx = floor(positive_rem(k, aux_num * n_samples[d]) / aux_num);

			Parameter_Mesh->at(k).sample_vec[d]  = aux_PM[aux_idx + aux_num_2].sample;
			Parameter_Mesh->at(k).Joint_PDF 	*= aux_PM[aux_idx + aux_num_2].PDF;

			aux_num *= n_samples[d];
			aux_num_2 += n_samples[d];
		}

	}
	return 0;
}

#endif