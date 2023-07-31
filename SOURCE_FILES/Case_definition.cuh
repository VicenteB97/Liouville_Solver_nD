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

//Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// This variable controls the size of the '.csv' files where the simulations are saved
#define MAX_FILE_SIZE_B 1024*1024*1024

// Choosing whether showing full or simplified timing information
#define OUTPUT_INFO true

#define TYPE float

// AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
#define TOLERANCE_AMR       0.0005
#define TOLERANCE_ConjGrad  powf(10,-6)
#define DISC_RADIUS         4.45

// State variables information
#define DIMENSIONS  2
#define DOMAIN_CTR  {0.00f, 0.00f}
#define DOMAIN_DIAM {12.0f, 12.0f}
inline const TYPE	IC_MEAN[DIMENSIONS] = {1.75f, 0.00f};	
inline const TYPE	IC_STD [DIMENSIONS] = {sqrtf(0.025f),sqrtf(0.025f)};	

// Vector field definition
// explanation: 
#define VEC_FIELD {X.dim[1], -2 * parameter.sample_vec[0] * X.dim[1] - X.dim[0] - parameter.sample_vec[1] * powf(X.dim[0], 3)}
#define DIVERGENCE -2 * parameter.sample_vec[0]

// Parameter information
#define PARAM_DIMENSIONS 2
inline const char   _DIST_NAMES	 [PARAM_DIMENSIONS] = {'N','N'};
inline const bool   _DIST_TRUNC	 [PARAM_DIMENSIONS] = {true,true};
inline const TYPE  _DIST_InfTVAL[PARAM_DIMENSIONS] = {0.0f, 0.0f};
inline const TYPE  _DIST_SupTVAL[PARAM_DIMENSIONS] = {1000.0f, 1000.0f};
inline TYPE 		_DIST_MEAN	 [PARAM_DIMENSIONS]	= {0.2f, 3.0f};
inline TYPE 		_DIST_STD	 [PARAM_DIMENSIONS]	= {sqrtf(0.02f),sqrtf(0.3f)};

// Impulse information
#define IMPULSE_TYPE 1
#if IMPULSE_TYPE == 1
	#define DELTA_JUMPS 3
	//	time | Imp | mean_vec  |   st. dev. | 	samples
	inline double 		D_JUMP_DIST_TIME 	[DELTA_JUMPS] 				= {0.6, 1.2, 2.4};
	inline const char   D_JUMP_DIST_NAMES	[DELTA_JUMPS * DIMENSIONS] 	= {'N','N','N','N','N','N'};
	inline const bool   D_JUMP_DIST_TRUNC	[DELTA_JUMPS * DIMENSIONS] 	= {true, true, true, true, true, true};
	inline const TYPE  D_JUMP_DIST_InfTVAL	[DELTA_JUMPS * DIMENSIONS] 	= {0, 0, 0, 0, 0, 0};
	inline const TYPE  D_JUMP_DIST_SupTVAL	[DELTA_JUMPS * DIMENSIONS] 	= {1000, 1000, 1000, 1000, 1000, 1000};
	inline TYPE 		D_JUMP_DIST_MEAN 	[DELTA_JUMPS * DIMENSIONS] 	= {0, 2.5, 0, 1, 0, 1.2};
	inline TYPE 		D_JUMP_DIST_STD  	[DELTA_JUMPS * DIMENSIONS] 	= {0, sqrtf(0.02f), 0, sqrtf(0.02f), 0, sqrtf(0.02f)};
	inline const int 	D_JUMP_DIST_SAMPLES [DELTA_JUMPS * DIMENSIONS] 	= {1, 20, 1, 20, 1, 20};
#endif

#define INCLUDE_XTRA_PARAMS false

#endif
