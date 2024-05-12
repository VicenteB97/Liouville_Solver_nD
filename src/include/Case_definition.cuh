
    #pragma once
    #define CASE "Duffing_1"

    // Choosing whether showing full or simplified timing information
    #define OUTPUT_INFO 0
    #define TERMINAL_INPUT_ALLOWED 0
    #define SAVING_TYPE "Y"
    
    #define FIRST_FRAME 0
    #define LAST_FRAME 0

    #define floatType float

    // AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
    #define TOLERANCE_AMR       5E-5
    #define TOLERANCE_ConjGrad  1E-9
    #define DISC_RADIUS         4.49

    // Phase space information
    #define PHASE_SPACE_DIMENSIONS  2
    #define DOMAIN_INF {-6, -6}
    #define DOMAIN_SUP {6, 6}
    #define FINEST_DISCR_LVL 9
    
    // Timing definitions:
    #define INIT_TIME 0
    #define FINAL_TIME 3
    #define TIME_STEP 0.01
    #define REINIT_STEPS 1
    #define SAVING_STEPS 4

    // Vector field definition
    // explanation: 
    #define VF_1 X.dim[1]
    #define D_1  0
    #define VF_2 -2 * parameter.sample_vec[0] * X.dim[1] - X.dim[0] - parameter.sample_vec[1] * powf(X.dim[0], 3)
    #define D_2  -2 * parameter.sample_vec[0]

    #define VEC_FIELD {VF_1, VF_2}
    #define DIVERGENCE D_1 + D_2

    static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = {'N', 'N'};
    static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = {true, true};
    static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_INF;
    static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = DOMAIN_SUP;
    static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = { 1.75f, 0.00f };
    static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = { sqrtf(0.015f),sqrtf(0.015f) };

    // Parameter information
    #define PARAM_SPACE_DIMENSIONS 2
    static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = { 'N','N' };
    static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = { true,true };
    static const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = { 0.0f, 0.0f };
    static const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = { 1000.0f, 1000.0f };
    static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = { 0.2f, 3.0f };
    static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = { sqrtf(0.002f),sqrtf(0.03f) };
    static floatType 		_DIST_N_SAMPLES[PARAM_SPACE_DIMENSIONS] = {10, 15};

        #define IMPULSE_TYPE 1
        #define DiracDelta_impulseCount 3
        //	time | Imp | mean_vec  |   st. dev. | 	samples
        static double 		deltaImpulse_distribution_TIME[DiracDelta_impulseCount] = {0.6, 1.2, 2.4};
        static const char   deltaImpulse_distribution_NAMES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {'N', 'N', 'N', 'N', 'N', 'N'};
        static const bool   deltaImpulse_distribution_isTRUNC[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { true, true, true, true, true, true };
        static const floatType  deltaImpulse_distribution_InfTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, 0, 0, 0, 0, 0 };
        static const floatType  deltaImpulse_distribution_SupTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 1000, 1000, 1000, 1000, 1000, 1000 };
        static floatType 		deltaImpulse_distribution_MEAN[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, 2.5, 0, 1, 0, 1.2 };
        static floatType 		deltaImpulse_distribution_STD[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 0, sqrtf(0.02f), 0, sqrtf(0.02f), 0, sqrtf(0.02f) };
        static const int 	deltaImpulse_distribution_SAMPLES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = { 1, 20, 1, 20, 1, 20 };

        #define INCLUDE_XTRA_PARAMS false
        