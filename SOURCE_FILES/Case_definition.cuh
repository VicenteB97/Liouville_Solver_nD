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
#define CASE "Mathieu equation"

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO 0

#define TYPE float     // only supported values are 'float' and 'double'


// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.00001
#define TOLERANCE_ConjGrad  powf(10,-8)     // RECOMMENDED: This appears to give good results...no need to change it
#define DISC_RADIUS         3.49

// State variables information
#define PHASE_SPACE_DIMENSIONS   2
#define DOMAIN_INF {-3,-3}
#define DOMAIN_SUP {3,3}

static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = { false, false };
static const TYPE   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = { 0, 0 };
static const TYPE   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = { 2, 2 };
static const TYPE	IC_MEAN[PHASE_SPACE_DIMENSIONS] = {0, -1};
static const TYPE	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.075),sqrtf(0.075) };

// Vector field definition (see the end of 'Classes.cuh' for the definition)
// explanation: p0 = a, p1 = q
#define VEC_FIELD_1    -X.dim[1]
#define DIVERGENCE_1   0

#define VEC_FIELD_2    -(parameter.sample_vec[0]-2*parameter.sample_vec[1]*cos(2*t))*X.dim[0]
#define DIVERGENCE_2   0

#define VEC_FIELD      {VEC_FIELD_1, VEC_FIELD_2}
#define DIVERGENCE      DIVERGENCE_1 + DIVERGENCE_2

// Parameter information
#define PARAM_SPACE_DIMENSIONS 2
static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = { 'U','U'};
static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = { false, false };
static const TYPE   _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 0,0 };
static const TYPE   _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 1,1 };
static TYPE 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 4,-1 };
static TYPE 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = { sqrtf(0.005), sqrtf(0.05) };

// Impulse parameter information (only delta or heaviside)
#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false

#endif