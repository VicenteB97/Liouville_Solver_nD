
    #pragma once
    #define CASE "SIR_r_5y5"

    // Choosing whether showing full or simplified timing information
    #define OUTPUT_INFO 0
    #define TERMINAL_INPUT_ALLOWED 0
    #define SAVING_TYPE "Y"
    
    #define FIRST_FRAME 0
    #define LAST_FRAME 0

    #define floatType float

    // AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
    #define TOLERANCE_AMR       1E-6
    #define TOLERANCE_ConjGrad  1E-9
    #define DISC_RADIUS         5.5

    // Phase space information
    #define PHASE_SPACE_DIMENSIONS  3
    #define DOMAIN_INF {0, 0, 0}
    #define DOMAIN_SUP {1, 1, 1}
    #define FINEST_DISCR_LVL 9
    
    // Timing definitions:
    #define INIT_TIME 0
    #define FINAL_TIME 31
    #define TIME_STEP 0.05
    #define REINIT_STEPS 2
    #define SAVING_STEPS 1

    // Use ad-hoc integrator? (ONLY FOR MATHIEU FOR NOW)
    #define SPECIAL_INTEGRATOR false

    // Vector field definition
    // explanation: 
    #define VF_1 (floatType)parameter.sample_vec[0] - (parameter.sample_vec[1] + parameter.sample_vec[2]*X.dim[1])*X.dim[0]
    #define D_1  -parameter.sample_vec[1] - parameter.sample_vec[2]*X.dim[1]
    #define VF_2 (floatType)(parameter.sample_vec[2]*X.dim[0] - parameter.sample_vec[1] - parameter.sample_vec[3]*X.dim[2])*X.dim[1]
    #define D_2  parameter.sample_vec[2]*X.dim[0] - parameter.sample_vec[1] - parameter.sample_vec[3]*X.dim[2]
    #define VF_3 (floatType)(parameter.sample_vec[3]*X.dim[1] - parameter.sample_vec[1])*X.dim[2]
    #define D_3  parameter.sample_vec[3]*X.dim[1] - parameter.sample_vec[1]

    #define VEC_FIELD {VF_1, VF_2, VF_3}
    #define DIVERGENCE (floatType)D_1 + D_2 + D_3

    static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = {'N', 'N', 'N'};
    static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = {true, true, true};
    static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_INF;
    static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_SUP;
    static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 0.75, 0.15, 0.10 };
    static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.0004),  sqrtf(0.0003), sqrtf(0.0002) };

    // Parameter information
    #define PARAM_SPACE_DIMENSIONS 4
    static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = {'U','U','N','G'};
    static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = {false, false, true, true};
    static const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = {0,0,0,0};
    static const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = {1,1,1,1};
    static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = {0.025,0.025,0.3,0.2};
    static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = {sqrtf(0.0005),sqrtf(0.0005),sqrtf(0.0001),sqrtf(0.0001)};
    static floatType 		_DIST_N_SAMPLES[PARAM_SPACE_DIMENSIONS] = {1,1,8,10};

            #define IMPULSE_TYPE 0
            #define INCLUDE_XTRA_PARAMS false