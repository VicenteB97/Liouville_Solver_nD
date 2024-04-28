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
	floatType sample_vec[DIM];
	floatType Joint_PDF;
};

class Param_pair {
public:
	floatType sample, PDF;
};

class Distributions{
public:
	floatType params[2];			// mean and variance. The appropriate choice of distribution parameters are given by the method of moments
	char Name;				// Distribution name. Currently supported distributions: Delta, Normal, Beta, Gamma and Uniform
	bool isTruncated;		// isTruncated? TRUE or FALSE
	floatType trunc_interval[2];	// Truncation interval (give min and max of the interval)
	intType  num_Samples;

	// Default constructor
	Distributions();
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
int16_t PDF_INITIAL_CONDITION(const Mesh& Mesh, thrust::host_vector<floatType>& PDF_value, const Distributions* IC_dist_parameters);


// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// ------------------ TO BUILD THE RANDOM PARAMETER MESH! ---------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //

/// @brief 
/// @param n_Samples 
/// @param PP 
/// @param Dist_params 
int16_t PARAMETER_VEC_BUILD(const int n_Samples, Param_pair* PP, const Distributions Dist_params);


template<uint16_t DIM>
__host__ __device__ Param_vec<DIM> Gather_Param_Vec(const uintType index, const Param_pair* Parameter_Array, const intType* n_Samples) {

	Param_vec<DIM> Output;

	Output.Joint_PDF = 1;

	uintType aux_samples_mult = 1;
	uintType aux_samples_sum = 0;


	for (uint16_t d = 0; d < DIM; d++) {
		uintType aux3 = n_Samples[d];
		uintType aux = floorf(positive_rem(index, aux3 * aux_samples_mult) / aux_samples_mult);

		Output.sample_vec[d] = Parameter_Array[aux + aux_samples_sum].sample;
		Output.Joint_PDF *= Parameter_Array[aux + aux_samples_sum].PDF;

		aux_samples_mult *= aux3;
		aux_samples_sum += aux3;
	}
	return Output;
}

/// @brief This function builds the Parameter Mesh that will be used in the Liouville Solver 
/// @param n_samples: Array where the number of samples per parameter is stored 
/// @param Parameter_Mesh: Parameter Mesh 
/// @param Dist_Parameters: Parameters' (hyper)parameters
/// @param Dist_Names: Distributions that will be assigned (N = Normal, U = Uniform, etc.)
/// @brief This function builds the Parameter Mesh that will be used in the Liouville Solver 
/// @param n_samples: Array where the number of samples per parameter is stored 
/// @param Parameter_Mesh: Parameter Mesh 
/// @param Dist_Parameters: Parameters' (hyper)parameters
/// @param Dist_Names: Distributions that will be assigned (N = Normal, U = Uniform, etc.)
template<uint16_t DIM>
int16_t RANDOMIZE(Param_pair* Parameter_Mesh,
	const Distributions* Dist_Parameters) {

	uintType aux = 0;

	for (uintType d = 0; d < DIM; d++) {
		intType nSamples = Dist_Parameters[d].num_Samples;

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