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
// Name of the case being simulated
#define CASE "Fitzhugh-Nagumo System"

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO true

#define floatType float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.00005
#define TOLERANCE_ConjGrad  powf(10,-7)
#define DISC_RADIUS         3.95

// State variables information
#define PHASE_SPACE_DIMENSIONS  2
#define DOMAIN_INF  {-2.5, -2.5}
#define DOMAIN_SUP  {2.5, 2.5}

inline const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = {0.1, 1.75};	
inline const floatType	IC_STD [PHASE_SPACE_DIMENSIONS] = {sqrtf(0.015f),sqrtf(0.015f)};

// Vector field definition
// explanation: p0 = R; p1 = I_{ext}; p2 = tau; p3 = a; p4 = b.
#define VEC_FIELD {X.dim[0] - powf(X.dim[0], 3) - X.dim[1] + parameter.sample_vec[0]*parameter.sample_vec[1], 1/parameter.sample_vec[2] * (X.dim[0] + parameter.sample_vec[3] - parameter.sample_vec[4] * X.dim[1])}
#define DIVERGENCE 1 - powf(X.dim[0],2) - parameter.sample_vec[4]/parameter.sample_vec[2]

// Parameter information
#define PARAM_SPACE_DIMENSIONS 5
inline const char   _DIST_NAMES	 [PARAM_SPACE_DIMENSIONS] = {'U','U','U','U','U'};
inline const bool   _DIST_isTRUNC	 [PARAM_SPACE_DIMENSIONS] = {false,false,false,false,false};
inline const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = {0,0,0,0,0};
inline const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = {1,1,1,1,1};
inline floatType 		_DIST_MEAN	 [PARAM_SPACE_DIMENSIONS]	= {0.1, 8, 15.5, 0.7, 0.8};
inline floatType 		_DIST_STD	 [PARAM_SPACE_DIMENSIONS]	= {sqrtf(0.001),sqrtf(0.1),sqrtf(0.2),sqrtf(0.001),sqrtf(0.01)};

#define IMPULSE_TYPE 0

#define INCLUDE_XTRA_PARAMS false

#endif