In this folder, we explain how to prepare the 'Case_definition.cuh' header file to perform your specified simulation.
Here, we give the general layout of the header file, but you can also see the other prepared examples for further ideas.

IMPORTANT NOTE: DO NOT CHANGE THE NAMES OF THE GLOBAL VARIABLES AND DEFINITIONS, ONLY ITS VALUES

1.- Adative Mesh Refinement tolerance
2.- Radius of RBF support as discretization multiples

3.- Spatial dimensions = Number of equations in the ODE System
4.- Domain Center
5.- Domain Diameter (per dimension, although rectangular mesh is still not implemented)
6.- Initial condition mean vector
7.- Initial condition std vector
8.- Vector field definition (X is the name for the state vectors. To access coordinate 'd', do 'X.dim[d]'. For the p-th parameter value, do 'parameter.sample_vec[p]')
9.- Vector field divergence (same applies as before)

10.- Parameter dimensions = Number of parameters in the ODE system (IC/BVs and parameters for delta-type impulses must not be included)
11.- Distribution names = Each entry is the distribution name for each parameter ('N' is normal, 'U' is uniform and 'G' is gamma)
11.- Distribution truncation = Each entry is the distribution truncation state for each parameter (true = truncated, false = not truncated)
12.- Infimum Truncation value = The infimum of the truncation interval for each parameter distribution
12.- Supremum Truncation value = The supremum of the truncation interval for each parameter distribution
13.- Each Mean value for every parameter
14.- Each std value for every parameter

NOTE 1: Only independent parameter distributions are considered. You CAN use correlated RVs as parameter distributions, but then they must be manually written in 'Parameter_distributions.cuh'

16.- If no impulses are to be included, set 'IMPULSE_TYPE' to 0.
17.- If we consider state-independent delta impulses, set 'IMPULSE_TYPE' to 1 and do:
    17.1.- Number of delta jumpCounts 
    17.2.- Each jumpCount's components' distribution type (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.3.- Each jumpCount's components' truncation logical (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.4.- Each jumpCount's components' truncation Inf. value (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.5.- Each jumpCount's components' truncation Sup. value (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.6.- Each jumpCount's components' truncation mean values (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.7.- Each jumpCount's components' truncation std values (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
    17.8.- Each jumpCount's components' truncation std samples (note there are jumpCountS * DIMENSION because we have a vector with PHASE_SPACE_DIMENSIONS coordinates: first PHASE_SPACE_DIMENSIONS entries are the components of 1st jumpCount)
18.- If we consider SQUARE Heaviside jumpCounts, set 'IMPULSE_TYPE' to 2 and do:
    18.1.- Number of SQUARE jumpCounts
    18.2.- Time values at which each impulse begins/ends ({begin, end, begin, end, ...})

For a complete example: 
    1D RDE: see '1_Linear_Eq.txt'
    With no jumps: see '4_VanDerPol_System.txt'
    With delta jumps: see '2_Duffing_System.txt'.
    With Heaviside jumps: see '3_Allelochemicals_System.txt'.

**IMPORTANT**: Currently (version 3.2.0 beta) does not support delta/heaviside impulses. Currently working on it.