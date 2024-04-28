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
//  Remember that you must also change the host system in the CMakeLists.txt file   //
// -------------------------------------------------------------------------------- //

// -----------------------------------------------------------------------//
// -------------------- MODIFY!!! ----------------------------------------//
// -----------------------------------------------------------------------//
// Name of the case being simulated
#define CASE "Mathieu equation"

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO 0

#define floatType float     // only supported values are 'float' and 'double'


// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       1.0E-7
#define TOLERANCE_ConjGrad  powf(10,-6)     // RECOMMENDED: This appears to give good results...no need to change it
#define DISC_RADIUS         4.49

// State variables information
#define PHASE_SPACE_DIMENSIONS   2
#define DOMAIN_INF {-7,-7}
#define DOMAIN_SUP {7,7}

static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = { 'N','N' };
static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = { false, false };
static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = { -3, -3};
static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = { 3, 3 };
static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = {0, -1};
static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrt((floatType)0.005),sqrt((floatType)0.005) };

// Vector field definition (see the end of 'Classes.cuh' for the definition)
// explanation: p0 = a, p1 = q
#define VEC_FIELD_1    X.dim[1]
#define DIVERGENCE_1   0

#define VEC_FIELD_2    -(parameter.sample_vec[0]-2*parameter.sample_vec[1]*cos(2*t))*X.dim[0]
#define DIVERGENCE_2   0

#define VEC_FIELD      {VEC_FIELD_1, VEC_FIELD_2}
#define DIVERGENCE      DIVERGENCE_1 + DIVERGENCE_2

// Parameter information
#define PARAM_SPACE_DIMENSIONS 2
static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = { 'U','U'};
static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = { false, false };
static const floatType   _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 3,-2 };
static const floatType   _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 5,0 };
static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 4,-3 };
static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = { sqrt((floatType)0.05), sqrt((floatType)0.005) };

// Impulse parameter information (only delta or heaviside)
#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false
