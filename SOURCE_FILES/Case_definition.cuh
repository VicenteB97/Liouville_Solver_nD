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
// -------------------------------------------------------------------------------- //

// -----------------------------------------------------------------------//
// -------------------- MODIFY!!! ----------------------------------------//
// -----------------------------------------------------------------------//
// Name of the case being simulated
#define CASE "VanDerPol System"

//Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// This variable controls the size of the '.csv' files where the simulations are saved
#define MAX_FILE_SIZE_B 1024*1024*1024

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO true

#define TYPE float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.00001
#define TOLERANCE_ConjGrad  powf(10,-6)
#define DISC_RADIUS         3.49

// State variables information
#define DIMENSIONS  2
#define DOMAIN_CTR  {0.00f, 0.00f}
#define DOMAIN_DIAM {11.0f, 11.0f}
// inline const TYPE	IC_MEAN[DIMENSIONS] = {0, 0};	
inline const TYPE	IC_MEAN[DIMENSIONS] = {0, 2.5};	
// inline const TYPE	IC_STD [DIMENSIONS] = {sqrtf(0.0015f),sqrtf(0.0015f)};
inline const TYPE	IC_STD [DIMENSIONS] = {sqrtf(0.015f),sqrtf(0.015f)};

// Vector field definition
// explanation: 
#define VF_1    X.dim[1]
#define D_1     0
#define VF_2    parameter.sample_vec[0]*(1 - powf(X.dim[0], 2))*X.dim[1] - X.dim[0]
#define D_2     parameter.sample_vec[0]*(1 - powf(X.dim[0], 2))

#define VEC_FIELD {VF_1, VF_2}
#define DIVERGENCE D_1 + D_2

// Parameter information
#define PARAM_DIMENSIONS 1
inline const char   _DIST_NAMES	 [PARAM_DIMENSIONS] = {'U'};
inline const bool   _DIST_TRUNC	 [PARAM_DIMENSIONS] = {true};
inline const TYPE  _DIST_InfTVAL[PARAM_DIMENSIONS] = {1.5};
inline const TYPE  _DIST_SupTVAL[PARAM_DIMENSIONS] = {3.5};
inline TYPE 		_DIST_MEAN	 [PARAM_DIMENSIONS]	= {2.5};
inline TYPE 		_DIST_STD	 [PARAM_DIMENSIONS]	= {sqrtf(0.1)};

#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false

#endif