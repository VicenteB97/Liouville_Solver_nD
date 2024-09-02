
    #pragma once
    #define CASE "VDP_1"

    // Choosing whether showing full or simplified timing information
    #define OUTPUT_INFO 0
    #define TERMINAL_INPUT_ALLOWED 0
    #define SAVING_TYPE "Y"
    
    #define FIRST_FRAME 0
    #define LAST_FRAME 0

    #define floatType float

    // AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
    #define TOLERANCE_AMR       1E-5
    #define TOLERANCE_ConjGrad  1E-9
    #define DISC_RADIUS         4.49

    // Phase space information
    #define PHASE_SPACE_DIMENSIONS  2
    #define DOMAIN_INF {-15, -15}
    #define DOMAIN_SUP {15, 15}
    #define FINEST_DISCR_LVL 10
    
    // Timing definitions:
    #define INIT_TIME 0
    #define FINAL_TIME 6
    #define TIME_STEP 0.01
    #define REINIT_STEPS 1
    #define SAVING_STEPS 4

    // Use ad-hoc integrator? (ONLY FOR MATHIEU FOR NOW)
    #define SPECIAL_INTEGRATOR false

    // Vector field definition
    // explanation:
    #define VF_1 (floatType)X.dim[1]
    #define D_1 0
    
    #define VF_2 (floatType)parameter.sample_vec[0]*(1 - powf(X.dim[0], 2))*X.dim[1] - parameter.sample_vec[1]*X.dim[0]
    #define D_2 parameter.sample_vec[0]*(1 - powf(X.dim[0], 2))
    
    #define VEC_FIELD {VF_1, VF_2}
    #define DIVERGENCE (floatType)D_1 + D_2

    static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = {'N', 'N'};
    static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = {true, true};
    static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_INF;
    static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_SUP;
    static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 1.0f, -1.0f };
    static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.09), sqrtf(0.09) };

    // Parameter information
    #define PARAM_SPACE_DIMENSIONS 1
    static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = {'N'};
    static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = {true};
    static const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = {-10.5};
    static const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = {10.5};
    static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = {1.0};
    static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = {sqrtf(0.09)};
    static floatType 		_DIST_N_SAMPLES[PARAM_SPACE_DIMENSIONS] = {50};

    #define IMPULSE_TYPE 0
    #define INCLUDE_XTRA_PARAMS false