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
#define CASE "SIR System"

// This variable controls the size of the '.csv' files where the simulations are saved
#define MAX_FILE_SIZE_B 1024*1024*1024

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO true

#define TYPE float     // only supported values are 'float' and 'double'

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.005
#define TOLERANCE_ConjGrad  powf(10,-6)     // RECOMMENDED: This appears to give good results...no need to change it
#define DISC_RADIUS         3.15

// State variables information
#define DIMENSIONS   3
#define DOMAIN_CTR  {0.5, 0.5, 0.5}
#define DOMAIN_DIAM {1, 1, 1}

inline const TYPE	IC_MEAN[DIMENSIONS] = {0.75, 0.15, 0.1};	
inline const TYPE	IC_STD [DIMENSIONS] = {sqrtf(0.0002), sqrtf(0.0002), sqrtf(0.0002)};

// Vector field definition
    // explanation: p0 = \Delta (birth rate), p1 = \mu (death rate), p2 = \beta (S to I transmission), p3 = \gamma (I to R transmission)
#define VEC_FIELD_1     parameter.sample_vec[0] - (parameter.sample_vec[1] + parameter.sample_vec[2]*X.dim[1])*X.dim[0]
#define DIVERGENCE_1   -parameter.sample_vec[1] - parameter.sample_vec[2]*X.dim[1]

#define VEC_FIELD_2    (parameter.sample_vec[2]*X.dim[0] - parameter.sample_vec[1] - parameter.sample_vec[3]*X.dim[2])*X.dim[1]
#define DIVERGENCE_2    parameter.sample_vec[2]*X.dim[0] - parameter.sample_vec[1] - parameter.sample_vec[3]*X.dim[2]

#define VEC_FIELD_3    (parameter.sample_vec[3]*X.dim[1] - parameter.sample_vec[1])*X.dim[2]
#define DIVERGENCE_3    parameter.sample_vec[3]*X.dim[1] - parameter.sample_vec[1]

#define VEC_FIELD      {VEC_FIELD_1, VEC_FIELD_2, VEC_FIELD_3}
#define DIVERGENCE      DIVERGENCE_1 + DIVERGENCE_2 + DIVERGENCE_3

// Parameter information
#define PARAM_DIMENSIONS 4
inline const char   _DIST_NAMES	 [PARAM_DIMENSIONS] = {'U','U','N','G'};
inline const bool   _DIST_TRUNC	 [PARAM_DIMENSIONS] = {false, false, true, true};
inline const TYPE  _DIST_InfTVAL[PARAM_DIMENSIONS] = {0,0,0,0};
inline const TYPE  _DIST_SupTVAL[PARAM_DIMENSIONS] = {1,1,1,1};
inline TYPE 		_DIST_MEAN	 [PARAM_DIMENSIONS]	= {0.025,0.025,0.3,0.2};
inline TYPE 		_DIST_STD	 [PARAM_DIMENSIONS]	= {sqrtf(0.0001),sqrtf(0.0001),sqrtf(0.0001),sqrtf(0.0001)};

// Impulse parameter information
#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false

#endif