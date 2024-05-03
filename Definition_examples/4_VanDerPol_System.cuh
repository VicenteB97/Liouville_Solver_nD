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

#define floatType float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       1E-5
#define TOLERANCE_ConjGrad  1E-9
#define DISC_RADIUS         5.49

// State variables information
#define PHASE_SPACE_DIMENSIONS  2
#define DOMAIN_INF {-6.0, -6.0}
#define DOMAIN_SUP {6.0, 6.0}

static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = { true, true };
// static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = { -2, 0.5 };
// static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = { 2, 4.5 };
static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_INF;
static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_SUP;
static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 0, 2.5 };
static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.02),sqrtf(0.02) };

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
static const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 0 };
static const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 5 };
static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 2.5 };
static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = { sqrtf(2) };

#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false
