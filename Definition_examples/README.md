# Creating experiments #

In this folder, we explain how to prepare the ```Case_definition.cuh``` header file to perform your specified simulation.
**NOTE** This file is located in ``` ./src/include ```.

This file is useful independently of the method used:

- You can create it in the ```.json``` file. Then ```setSimulator.py``` to automatically generate the file and build, compile and run the example(s).

- Manually in the ```Case_definition.cuh``` file for each experiment.

Here, we give the general layout of the header file, but you can also see the other prepared examples for further ideas.

**IMPORTANT NOTE**: DO NOT CHANGE THE NAMES OF THE GLOBAL VARIABLES AND DEFINITIONS, ONLY ITS VALUES

1. **Saving** To decide the saving of the frames (iterations) of the simulation:
    1. Type: Can be ```N``` (No), ```Y``` (Yes/All) or ```P``` (Partial).
    2. In the case of ```P```, we need to set the initial frame ```first_frame``` and the final frame ```last_frame```.

2. **Simulation parameters** This family of parameters contain the tolerances, domain definition and timing information:
    1. ```floatType```: Floating-point type (```float``` or ```double```).
    2. ```TOLERANCE_AMR```: Adaptive Mesh Refinement tolerance.
    3. ```TOLERANCE_ConjGrad```: Conjugate Gradient tolerance.
    4. ```DISC_RADIUS```: The radial basis function radius as a multiple of the spatial discretization.
    5. ```phase_space_dimensions```: Number of dimensions of the phase space.
    6. ```DOMAIN_INF```: Lowest corner of the phase space domain.
    7. ```DOMAIN_SUP```: Highest corner of the phase space domain.
    8. ```param_space_dimensions```: Number of parameters/dimension of the parameter space.
    9. ```disc_finest_level```: Log_2 number of points of the finest level for AMR. 
    10. ```t0```: Initial time of simulation.
    11. ```tF```: Final time of simulation.
    12. ```delta_t```: Integrator time-step.
    13. ```reinit_steps```: Number of integrator steps for reinitialization.
    14. ```saving_steps```: Number of reinitialization steps for saving frame.

3. **Vector field and divergence** This family of parameters define the vector field that evolves the initial condition. Also the divergence field, which is needed for the characteristic equation method:
    1. ```VF_d```: Denotes the *d*-th component of the vector field.
    2. ```D_d```: The *d*-th term of the divergence term; equivalently, the *d*-th summand of the jacobian matrix trace.
    3. ```VEC_FIELD```: This term's general input structure ***MUST NOT BE CHANGED***, only add the corresponding terms; e.g.:
        - If there is only one equation: ```"VEC_FIELD": "{VF_1}"```.
        - If there are two equations: ```"VEC_FIELD": "{VF_1, VF_2}"```.
        - If there are three equations: ```"VEC_FIELD": "{VF_1, VF_2, VF_3}"```.
    4. ```DIVERGENCE```: Same idea as before, just add the corresponding terms into the sum.

4. **Initial condition** This family of parameters define the distribution information on the initial condition function.
    1. ```IC_NAMES```: This is a set with the name of the distribution for each component. Can be ```N``` (= Normal/Gaussian), ```G``` (= Gamma) or ```B``` (= Beta).
        - ***NOTE*** Non-smooth functions are not currently supported as initial conditions.
    2. ```IC_isTRUNC```: Defines whether the distribution is truncated. As every computational experiment, the distribution will be truncated, however, setting this value to true will allow you to truncate it in the desired domain. If false, it automatically truncates in an interval that is centered at the mean and an 8 standard deviation radius (further restrictions obviously apply for gamma and beta distributions).
    3. ```IC_InfTVAL```: Infimum of the custom truncation interval. Can be set to the domain boundaries by writing ```IC_InfTVAL = DOMAIN_INF```.
    4. ```IC_SupTVAL```: Supremum of the custom truncation interval. Can be set to the domain boundaries by writing ```IC_SupTVAL = DOMAIN_SUP```.
    5. ```IC_MEAN```: Distributions' expectation/mean.
    6. ```IC_STD```: Distributions' standard deviation.

    ***NOTE 1***: Only independent parameter distributions are considered. You CAN use correlated RVs, but then they must be manually written in code. Contact the author.
    ***NOTE 2***: Suitable parameters for non-gaussian distributions are obtained by the so-called method of moments; that way we always define the distributions by their mean and variance. The appropriate parameters are computed in the background.

5. **Parameter distributions** This family of parameters define the distribution information on the initial condition function.
    1. ```IC_NAMES```: This is a set with the name of the distribution for each component. Can be ```N``` (= Normal/Gaussian), ```G``` (= Gamma), ```B``` (= Beta) or ```U``` (= Uniform).
    2. ```IC_isTRUNC```: Defines whether the distribution is truncated. As every computational experiment, the distribution will be truncated, however, setting this value to true will allow you to truncate it in the desired domain. If false, it automatically truncates in an interval that is centered at the mean and an 8 standard deviation radius (further restrictions obviously apply for gamma and beta distributions).
    3. ```IC_InfTVAL```: Infimum of the custom truncation interval.
    4. ```IC_SupTVAL```: Supremum of the custom truncation interval.
    5. ```IC_MEAN```: Distributions' expectation/mean.
    6. ```IC_STD```: Distributions' standard deviation.
    7. ```DIST_N_SAMPLES```: Samples for the distribution. These samples are not randomly chosen. Actually, these values determine the number of points in the discretization of the parameters' density support. These values and their density weights are used for the ensemble mean PDF computation.

***NOTE***: The following instructions are to be revised soon. There are no guarantees that the following will work:

6. If no impulses are to be included, set 'IMPULSE_TYPE' to 0.
7. If we consider state-independent delta impulses, set 'IMPULSE_TYPE' to 1 and do:
    17.1.- Number of delta jumpCounts 
    17.2.- Each jumpCount's components' distribution type (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.3.- Each jumpCount's components' truncation logical (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.4.- Each jumpCount's components' truncation Inf. value (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.5.- Each jumpCount's components' truncation Sup. value (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.6.- Each jumpCount's components' truncation mean values (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.7.- Each jumpCount's components' truncation std values (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.8.- Each jumpCount's components' truncation std samples (note there are ```jumpCountS * DIMENSION``` because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
8. If we consider SQUARE Heaviside jumpCounts, set 'IMPULSE_TYPE' to 2 and do:
    18.1.- Number of SQUARE jumpCounts
    18.2.- Time values at which each impulse begins/ends ({begin, end, begin, end, ...})

For a complete example: 
    1D RDE: see '1_Linear_Eq.txt'
    With no jumps: see '4_VanDerPol_System.txt'
    With delta jumps: see '2_Duffing_System.txt'.
    With Heaviside jumps: see '3_Allelochemicals_System.txt'.

**IMPORTANT**: Currently (version 4.0.0 beta) does not support delta/heaviside impulses. Currently working on it.
