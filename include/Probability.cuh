//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
// This part contains the definitions of the functions that will be used to build the parameter mesh + j-PDF   //
// We STRONGLY RECOMMEND not changing absolutely anything if your only use of this program is to simulate	   //
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//

#ifndef __PROBABILITY_CUH__
#define __PROBABILITY_CUH__

#include "headers.cuh"
#include "Domain.cuh"
#include "utils/numeric_defs.cuh"

// These are ONLY used here
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/beta.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<uint16_t DIM>
class Param_vec{
public:
	TYPE sample_vec[DIM];
	TYPE Joint_PDF;
};

class Param_pair {
public:
	TYPE sample, PDF;
};

class Distributions{
public:
	TYPE params[2];			// mean and variance. The appropriate choice of distribution parameters are given by the method of moments
	char Name;				// Distribution name. Currently supported distributions: Delta, Normal, Beta, Gamma and Uniform
	bool isTruncated;		// isTruncated? TRUE or FALSE
	TYPE trunc_interval[2];	// Truncation interval (give min and max of the interval)
	INT  num_Samples;

	// Default constructor
	Distributions() {
		params[0] = 0;
		params[1] = 0;

		Name = 'D';

		isTruncated = true;

		trunc_interval[0] = 0;
		trunc_interval[1] = 0;

		num_Samples = 1;
	}
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief This function creates the IC PDF. We assume it is the tensor product of Normal Distributions
/// @param Points_per_dimension 
/// @param Mesh 
/// @param PDF_value 
/// @param IC_dist_parameters 
int16_t PDF_INITIAL_CONDITION(const Mesh& Mesh, thrust::host_vector<TYPE>& PDF_value, const Distributions *IC_dist_parameters) {

	std::vector<TYPE> temp_val(Mesh.Nodes_per_Dim * PHASE_SPACE_DIMENSIONS);

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		// Create the arrays for each dimension!
		TYPE expectation = IC_dist_parameters[d].params[0], std_dev = IC_dist_parameters[d].params[1],
			x0 = fmax(Mesh.Boundary_inf.dim[d], expectation - 8 * std_dev), 
			xF = fmin(Mesh.Boundary_sup.dim[d], expectation + 8 * std_dev), 
			rescale_CDF = 1;

		if(IC_dist_parameters[d].Name == 'N' || IC_dist_parameters[d].Name == 'n') {
			auto dist = boost::math::normal_distribution(expectation, std_dev);

			if (IC_dist_parameters[d].isTruncated) {
				x0 = fmaxf(x0, IC_dist_parameters[d].trunc_interval[0]);
				xF = fminf(xF, IC_dist_parameters[d].trunc_interval[1]);

				rescale_CDF = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);
			}

			#pragma omp parallel for
			for (INT k = Mesh.Nodes_per_Dim * d; k < Mesh.Nodes_per_Dim * (d + 1); k++) {
				temp_val[k] = boost::math::pdf(dist, Mesh.Get_node((k - Mesh.Nodes_per_Dim * d) * pow(Mesh.Nodes_per_Dim, d)).dim[d]) / rescale_CDF;
			}
		}
	}

	std::cout << "Filling initial density...\n";

	#pragma omp parallel for
	for (INT k = 0; k < Mesh.Total_Nodes(); k++) {
		TYPE val = 1;
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT temp_idx = floor(positive_rem(k, pow(Mesh.Nodes_per_Dim, d + 1)) / pow(Mesh.Nodes_per_Dim, d));

			val *= temp_val[temp_idx + Mesh.Nodes_per_Dim * d];
		}
		PDF_value[k] = val;
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

		
		if(Dist_params.isTruncated){
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			TYPE rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// Mesh discretization
			dx = (xF - x0) / (n_Samples - 1);

			#pragma omp parallel for
			for (int i = 0; i < n_Samples; i++) {
				TYPE x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (n_Samples - 1);

		#pragma omp parallel for
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

		#pragma omp parallel for
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

		
		if(Dist_params.isTruncated){
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			TYPE rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// Mesh discretization
			dx = (xF - x0) / (n_Samples - 1);

			#pragma omp parallel for
			for (int i = 0; i < n_Samples; i++) {
				TYPE x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (n_Samples - 1);

		#pragma omp parallel for
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

		
		if(Dist_params.isTruncated){
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			TYPE rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// Mesh discretization
			dx = (xF - x0) / (n_Samples - 1);

			#pragma omp parallel for
			for (int i = 0; i < n_Samples; i++) {
				TYPE x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (n_Samples - 1);
		
		#pragma omp parallel for
		for (int i = 0; i < n_Samples; i++) {
			TYPE x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	std::cout << "Parameter distribution not recognized. Choose correctly please. \nExiting program...\n";
	return -1;
}

template<uint16_t DIM>
__host__ __device__ inline Param_vec<DIM> Gather_Param_Vec(const UINT index, const Param_pair* Parameter_Array, const INT* n_Samples){

	Param_vec<DIM> Output;

	Output.Joint_PDF = 1;

	UINT aux_samples_mult = 1;
	UINT aux_samples_sum  = 0;

	
	for (uint16_t d = 0; d < DIM; d++){
		UINT aux3 = n_Samples[d];
		UINT aux = floorf(positive_rem(index, aux3 * aux_samples_mult) / aux_samples_mult );

		Output.sample_vec[d] = Parameter_Array[aux + aux_samples_sum].sample;
		Output.Joint_PDF 	*= Parameter_Array[aux + aux_samples_sum].PDF;

		aux_samples_mult *= aux3;
		aux_samples_sum  += aux3;
	}
	return Output;
}

/// @brief This function builds the Parameter Mesh that will be used in the Liouville Solver 
/// @param n_samples: Array where the number of samples per parameter is stored 
/// @param Parameter_Mesh: Parameter Mesh 
/// @param Dist_Parameters: Parameters' (hyper)parameters
/// @param Dist_Names: Distributions that will be assigned (N = Normal, U = Uniform, etc.)
template<uint16_t DIM>
int16_t RANDOMIZE(Param_pair* 			Parameter_Mesh, 
				const Distributions		*Dist_Parameters) {

	UINT aux = 0;
	
	for (UINT d = 0; d < DIM; d++){
		INT nSamples = Dist_Parameters[d].num_Samples;

		// call the parameter pair vec. function
		Param_pair* PP = new Param_pair[nSamples];

		errorCheck(PARAMETER_VEC_BUILD(nSamples, PP, Dist_Parameters[d]));

		std::copy_n(&PP[0], nSamples, &Parameter_Mesh[aux]);

		delete[] PP;
		aux += nSamples;
	}

	return 0;
}

#endif