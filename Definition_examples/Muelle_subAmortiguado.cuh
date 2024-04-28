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
// 
// -------------------------------------------------------------------------------- //
//  HERE ARE THE DEFINITIONS THAT CAN BE CHANGED ACCORDING TO THE PROBLEM TO STUDY  //
// -------------------------------------------------------------------------------- //

// -----------------------------------------------------------------------//
// -------------------- MODIFY!!! ----------------------------------------//
// -----------------------------------------------------------------------//
// Name of the case being simulated
#define CASE "Duffing System"

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO 2

#define floatType float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.00005
#define TOLERANCE_ConjGrad  powf(10,-8)
#define DISC_RADIUS         3.45

// State variables information
#define PHASE_SPACE_DIMENSIONS 2
#define DOMAIN_INF	{-3.5, -3.5}
#define DOMAIN_SUP	{3.5, 3.5}

static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = { false, false };
static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = { -6, -6 };
static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = { 6, 6 };
static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 1.75f, 0.00f };
static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.015f),sqrtf(0.015f) };



// Vector field definition
// explanation: 
#define VEC_FIELD {X.dim[1], -parameter.sample_vec[0]/parameter.sample_vec[1] * X.dim[0] -parameter.sample_vec[2]/parameter.sample_vec[1]* X.dim[1] }
#define DIVERGENCE -2 * parameter.sample_vec[0]

// Parameter information
#define PARAM_SPACE_DIMENSIONS 3
static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = { 'N','N','N'};
static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = { true,true ,true};
static const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 0.0f, 0.0f,0.0f };
static const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 1000.0f, 1000.0f,1000.0f };
static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 0.9f, 1.0f,0.3f};
static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = {1.0f,0.0f,0.0f };

// Impulse information
#define IMPULSE_TYPE 1
#if IMPULSE_TYPE == 1
#define DiracDelta_impulseCount 1
//	time | Imp | mean_vec  |   st. dev. | 	samples
static double 		deltaImpulse_distribution_TIME[DiracDelta_impulseCount] = {0};
static const char   deltaImpulse_distribution_NAMES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {'N','N'};
static const bool   deltaImpulse_distribution_isTRUNC[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { true, true };
static const floatType   deltaImpulse_distribution_InfTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, 0};
static const floatType   deltaImpulse_distribution_SupTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 1000, 1000};
static floatType 		deltaImpulse_distribution_MEAN[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, 3};
static floatType 		deltaImpulse_distribution_STD[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, sqrtf(0.02f) };
static const int 	deltaImpulse_distribution_SAMPLES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 1, 1};
#endif

#define INCLUDE_XTRA_PARAMS false
