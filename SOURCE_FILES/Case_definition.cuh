//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//														//
//		Written by: Vicente José Bevia Escrig			//
//		Mathematics Ph.D. student (2020-2024) at:		//
//		Instituto de Matemática Multidisciplinar,		//
//		Universitat Politècnica de València, Spain		//
//														//
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

#ifndef __CASE_DEFINITION_CUH__
#define __CASE_DEFINITION_CUH__

// -------------------------------------------------------------------------------- //
//  HERE ARE THE DEFINITIONS THAT CAN BE CHANGED ACCORDING TO THE PROBLEM TO STUDY  //
//  Remember that you must also change the host system in the CMakeLists.txt file   //
// -------------------------------------------------------------------------------- //

// -----------------------------------------------------------------------//
// -------------------- MODIFY!!! ----------------------------------------//
// -----------------------------------------------------------------------//
// Name of the case being simulated
#define CASE "Compartmental model"

//Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// This variable controls the size of the files where the simulations are saved
#define MAX_FILE_SIZE_B 2*1024*1024*1024

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO true

#define TYPE float     // only supported values are 'float' and 'double'

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.0005
#define TOLERANCE_ConjGrad  powf(10,-6)     // RECOMMENDED: This appears to give good results...no need to change it
#define DISC_RADIUS         3.49

// State variables information
#define DIMENSIONS   2
#define DOMAIN_CTR  {1,1}
#define DOMAIN_DIAM {2,2}

static const TYPE	IC_MEAN[DIMENSIONS] = {1.2, 1.2};
static const TYPE	IC_STD[DIMENSIONS] = { sqrtf(0.00075),sqrtf(0.00075) };

// Vector field definition (see the end of 'Classes.cuh' for the definition)
// explanation: p0 = a_1, p1 = a_2, p2 = alpha, p3 = beta
#define VEC_FIELD_1    -parameter.sample_vec[0] * X.dim[0] + parameter.sample_vec[2] * powf(sinf(M_PI/4 * t),2)
#define DIVERGENCE_1   -parameter.sample_vec[0]

#define VEC_FIELD_2     parameter.sample_vec[0] * X.dim[0] - parameter.sample_vec[1] * X.dim[1] + parameter.sample_vec[3]*t*t*t/(expf(t) + 1)
#define DIVERGENCE_2   -parameter.sample_vec[1]

#define VEC_FIELD      {VEC_FIELD_1, VEC_FIELD_2}
#define DIVERGENCE      DIVERGENCE_1 + DIVERGENCE_2

// Parameter information
#define PARAM_DIMENSIONS 4
static const char   _DIST_NAMES[PARAM_DIMENSIONS] = { 'B','B','G','G' };
static const bool   _DIST_TRUNC[PARAM_DIMENSIONS] = { false, false, true, true };
static const TYPE   _DIST_InfTVAL[PARAM_DIMENSIONS] = { 0,0,0,0 };
static const TYPE   _DIST_SupTVAL[PARAM_DIMENSIONS] = { 1,1,1,1 };
static TYPE 		_DIST_MEAN[PARAM_DIMENSIONS] = { 0.2,0.35,0.1,0.2 };
static TYPE 		_DIST_STD[PARAM_DIMENSIONS] = { sqrtf(0.005),sqrtf(0.005),sqrtf(0.002),sqrtf(0.002) };

// Impulse parameter information (only delta or heaviside)
#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false

#endif