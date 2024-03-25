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
#define CASE "Duffing System"

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO 0

#define TYPE float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.00005
#define TOLERANCE_ConjGrad  powf(10,-8)
#define DISC_RADIUS         3.45

// State variables information
#define PHASE_SPACE_DIMENSIONS 2
#define DOMAIN_INF	{-6, -6}
#define DOMAIN_SUP	{6, 6}

static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = { false, false };
static const TYPE   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = { -6, -6 };
static const TYPE   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = { 6, 6 };
static const TYPE	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 1.75f, 0.00f };
static const TYPE	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.015f),sqrtf(0.015f) };



// Vector field definition
// explanation: 
#define VEC_FIELD {X.dim[1], -2 * parameter.sample_vec[0] * X.dim[1] - X.dim[0] - parameter.sample_vec[1] * powf(X.dim[0], 3)}
#define DIVERGENCE -2 * parameter.sample_vec[0]

// Parameter information
#define PARAM_SPACE_DIMENSIONS 2
static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = { true,true };
static const TYPE  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 0.0f, 0.0f };
static const TYPE  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 1000.0f, 1000.0f };
static TYPE 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 0.2f, 3.0f };
static TYPE 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = { sqrtf(0.002f),sqrtf(0.03f) };

// Impulse information
#define IMPULSE_TYPE 1
#if IMPULSE_TYPE == 1
#define DiracDelta_impulseCount 3
//	time | Imp | mean_vec  |   st. dev. | 	samples
static double 		deltaImpulse_distribution_TIME[DiracDelta_impulseCount] = { 0.6, 1.2, 2.4 };
static const char   deltaImpulse_distribution_NAMES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 'N','N','N','N','N','N' };
static const bool   deltaImpulse_distribution_isTRUNC[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { true, true, true, true, true, true };
static const TYPE  deltaImpulse_distribution_InfTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, 0, 0, 0, 0, 0 };
static const TYPE  deltaImpulse_distribution_SupTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 1000, 1000, 1000, 1000, 1000, 1000 };
static TYPE 		deltaImpulse_distribution_MEAN[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, 2.5, 0, 1, 0, 1.2 };
static TYPE 		deltaImpulse_distribution_STD[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, sqrtf(0.02f), 0, sqrtf(0.02f), 0, sqrtf(0.02f) };
static const int 	deltaImpulse_distribution_SAMPLES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 1, 20, 1, 20, 1, 20 };
#endif

#define INCLUDE_XTRA_PARAMS false

#endif