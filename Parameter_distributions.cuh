#ifndef __PARAMETER_DISTRIBUTIONS_CUH__
#define __PARAMETER_DISTRIBUTIONS_CUH__


#include "Classes.cuh"

// THERE ARE GOING TO BE PARAMETER_SPACE_DIMENSIONS + 1 PARAMETER DISTRIBUTIONS

// TO BE CHANGED: PUT THE DOMAIN INFORMATION IN A DISTRIBUTION-INDEPENDENT FUNCTION

// --------------------- INITIAL CONDITION -------------------------------------------
// -----------------------------------------------------------------------------------
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
			PDF_value[i + Points_per_dimension * j] =	1 / sqrt(2 * M_PI * var * var) *   exp(-0.5 / (var * var)   * pow(Mesh[i + Points_per_dimension * j].position[0] - mean1, 2)) *
														1 / sqrt(2 * M_PI * var2 * var2) * exp(-0.5 / (var2 * var2) * pow(Mesh[i + Points_per_dimension * j].position[1] - mean2, 2)); // Normal PDF
		}
	}
}

// --------------------- MODEL PARAMETERS --------------------------------------------
// -----------------------------------------------------------------------------------
void PDF_PARAM_1(int n_Samples, Param_pair* PP_1, double* Dist_params) {
	// In our case, this parameter is: mu
	// Distribution: Normal (mean = 1.1; var = 1.1*0.05)

	double expectation = Dist_params[0], var = Dist_params[1];

	if (n_Samples == 1) {
		*PP_1 = { expectation, 1 };
	}
	else {

		double x0 = fmax(expectation - 5 * sqrt(var), 0);
		double xF = expectation + 5 * sqrt(var);

		double dx = (xF - x0) / (n_Samples - 1);

		for (int i = 0; i < n_Samples; i++) {
			PP_1[i] = { x0 + i * dx, 1 / sqrt(2 * M_PI * var) * exp(-0.5 / var * pow(x0 + i * dx - expectation, 2)) };
		}
	}
}

void PDF_PARAM_2(int n_Samples, Param_pair* PP_2, double* Dist_params) {
	// In our case, this parameter is: A
	// Distribution: Uniform (min = 0.025; max = 0.055)

	double expectation = Dist_params[0], var = Dist_params[1];

	if (n_Samples == 1) {
		*PP_2 = { expectation,1 };
	}
	else {
		double x0 = fmax(expectation - 5 * sqrt(var), 0);
		double xF = expectation + 5 * sqrt(var);

		double dx = (xF - x0) / (n_Samples - 1);

		for (int i = 0; i < n_Samples; i++) {
			PP_2[i] = { x0 + i * dx, 1 / sqrt(2 * M_PI * var) * exp(-0.5 / var * pow(x0 + i * dx - expectation, 2)) };
		}
	}
}


// --------------------- RANDOMIZE FUNCTION ------------------------------------------
// -----------------------------------------------------------------------------------
void RANDOMIZE(const int* n_samples, const int Total_Samples, std::vector<Param_vec>* Parameter_Mesh, double* Dist_Parameters) {

	Param_pair* PP_1 = new Param_pair[n_samples[0]];
	Param_pair* PP_2 = new Param_pair[n_samples[1]];

	double aux_params_1[2] = { Dist_Parameters[0], Dist_Parameters[1] };
	double aux_params_2[2] = { Dist_Parameters[2], Dist_Parameters[3] };

	PDF_PARAM_1(n_samples[0], PP_1, aux_params_1);
	PDF_PARAM_2(n_samples[1], PP_2, aux_params_2);

	for (int i = 0; i < Total_Samples; i++) {
		int i1 = i % n_samples[0];
		int i2 = (int)floor(i / n_samples[0]);

		Parameter_Mesh->push_back({ PP_1[i1].sample, PP_2[i2].sample , PP_1[i1].PDF * PP_2[i2].PDF });
	}

	delete[] PP_1;
	delete[] PP_2;
}

#endif