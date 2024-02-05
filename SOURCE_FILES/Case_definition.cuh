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

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO 0

#define TYPE float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.0001
#define TOLERANCE_ConjGrad  powf(10,-8)
#define DISC_RADIUS         3.49

// State variables information
#define PHASE_SPACE_DIMENSIONS  2
#define DOMAIN_INF {-5.5, -5.5}
#define DOMAIN_SUP {5.5 , 5.5}

static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = { true, true };
static const TYPE   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = { -2, 0.5 };
static const TYPE   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = { 2, 4.5 };
static const TYPE	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 0, 2.5 };
static const TYPE	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.015f),sqrtf(0.015f) };

// Vector field definition
// explanation: 
#define VF_1    X.dim[1]
#define D_1     0
#define VF_2    parameter.sample_vec[0]*(1 - powf(X.dim[0], 2))*X.dim[1] - X.dim[0]
#define D_2     parameter.sample_vec[0]*(1 - powf(X.dim[0], 2))

#define VEC_FIELD {VF_1, VF_2}
#define DIVERGENCE D_1 + D_2

// Parameter information
#define PARAM_SPACE_DIMENSIONS 1
static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = { 'U' };
static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = { true };
static const TYPE  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 1.5 };
static const TYPE  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 3.5 };
static TYPE 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 2.5 };
static TYPE 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = { sqrtf(0.1) };

#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false

#endif