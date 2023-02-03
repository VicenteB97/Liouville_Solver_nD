#ifndef __PARAMETER_DISTRIBUTIONS_CUH__
#define __PARAMETER_DISTRIBUTIONS_CUH__


#include "Classes.cuh"



// PUT THE FUNCTIONS WHERE WE DO THE DISTRIBUTION ASSIGNMENTS HEREEEEE



// WE CAN EASILY IMPLEMENT DIFFERENT 1D-DISTRIBUTIONS

/// @brief This function computes the value given by the PDF of a Normal Distribution at input, with mean and variance as parameters
/// @tparam T Select the type/class to be used
/// @param input 
/// @param mean 
/// @param variance 
/// @return 
template<class T>
inline T Normal_Dist(T input, T mean, T variance){
	return 1 / sqrt(2 * M_PI * variance) * exp(-0.5 / variance * pow(input - mean, 2));
}

/// @brief This function computes the value given by the PDF of a Uniform Distribution at input, with inf. and sup. of the support interval parameters
/// @tparam T 
/// @param input Where we compute
/// @param x0 Inf. of the support interval
/// @param xF Sup. of the support interval
/// @return 
template<class T>
inline T Unif_Dist(T input, T x0, T xF){
	T mid_Pt = 0.5*(xF + x0), supp_size = xF - x0;

	if (abs(input - mid_Pt) < supp_size/2){
		return (input - x0) / (xF - x0);
	}
	else{ return (T)0;}
}


// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// --------------------- TO BUILD THE INITIAL CONDITION! ----------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //

// THIS FUNCTION REMAINS TO BE GENERALIZED FOR THE DIMENSION INDEPENDENT SIMULATIONS
void PDF_INITIAL_CONDITION(int Points_per_dimension, const gridPoint* Mesh, thrust::host_vector<double>& PDF_value, double* IC_dist_parameters) {
	// read parameters:
	// Mean and var of the 1st variable
	double mean1 = IC_dist_parameters[0];
	double mean2 = IC_dist_parameters[2];

	// Mean and var of the 2nd variable
	double var  = IC_dist_parameters[1];
	double var2 = IC_dist_parameters[3];

	// build the IC pts per dimension
	for (int j = 0; j < Points_per_dimension; j++) {
		for (int i = 0; i < Points_per_dimension; i++) {
			PDF_value[i + Points_per_dimension * j] =	Normal_Dist<double>(Mesh[i + Points_per_dimension * j].dim[0], mean1, var*var) * 
														Normal_Dist<double>(Mesh[i + Points_per_dimension * j].dim[1], mean2, var2*var2);
		}
	}
}


// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// ------------------ TO BUILD THE RANDOM PARAMETER MESH! ---------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
void PARAMETER_VEC_BUILD(const int n_Samples, Param_pair* PP, const double* Dist_params, const char Dist_Names) {

	double expectation = Dist_params[0], var = Dist_params[1];

	if (n_Samples == 1) {
		*PP = { expectation, 1 };
	}
	else {

		if (Dist_Names == 'N'){
			double x0 = fmax(expectation - 5 * sqrt(var), 0);
			double xF = expectation + 5 * sqrt(var);

			double dx = (xF - x0) / (n_Samples - 1);

			for (int i = 0; i < n_Samples; i++) {
				PP[i] = { x0 + i * dx, Normal_Dist<double>(x0 + i * dx, expectation, var) }; // other distributions could be used
			}
		}
		else if(Dist_Names == 'U'){

			double x0 = Dist_params[0];
			double xF = Dist_params[1];

			double dx = (xF - x0) / (n_Samples - 1);
			for (int i = 0; i < n_Samples; i++) {
				PP[i] = { x0 + i * dx, Unif_Dist<double>(x0 + i * dx, x0, xF) }; // other distributions could be used
			}
		}
	}
}

/// @brief This function builds the Parameter Mesh that will be used in the Liouville Solver 
/// @param n_samples Array where the number of samples per parameter is stored 
/// @param Total_Samples Total number of samples
/// @param Parameter_Mesh Parameter Mesh 
/// @param Dist_Parameters Parameters' (hyper)parameters
/// @param Dist_Names Distributions that will be assigned (N = Normal, U = Uniform, etc.)
void RANDOMIZE(	const int* 				n_samples, 
				const int 				Total_Samples, 
				std::vector<Param_vec>* Parameter_Mesh, 
				const double* 			Dist_Parameters,
				const char* 			Dist_Names) {

	std::vector<Param_pair> aux_PM;

	for (unsigned int d = 0; d < PARAM_DIMENSIONS; d++){
		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[n_samples[d]];
		double aux_Dist_Parameters[2] = {(double) Dist_Parameters[2*d], (double) Dist_Parameters[2*d + 1]};

		PARAMETER_VEC_BUILD(n_samples[d], PP, aux_Dist_Parameters, Dist_Names[d]);

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
		Parameter_Mesh->at(k).Joint_PDF = aux_PM[aux_idx].PDF;

		for (unsigned int d = 1; d < PARAM_DIMENSIONS; d++){

			aux_idx = floor(positive_rem(k, aux_num * n_samples[d]) / aux_num);

			Parameter_Mesh->at(k).sample_vec[d]  = aux_PM[aux_idx + aux_num_2].sample;
			Parameter_Mesh->at(k).Joint_PDF 	*= aux_PM[aux_idx + aux_num_2].PDF;

			aux_num *= n_samples[d];
			aux_num_2 += n_samples[d];
		}

	}
	
}

#endif