# Liouville_Solver_nD
In this repository we have a CUDA solution for the computation of the solution to the Liouville Equation in general dimensions.
We will explain the steps so you can familiarize with the code and test it for yourself! (Any optimizations/errors, please let me know)

REQUIREMENTS:
Ubuntu or WSL (for Windows)
g++ compiler (built in g++ 11.3.0)
gcc compiler (built in gcc 11.3.0)
nvcc compiler (built in nvcc 11.5) -- Obviously you need a CUDA-Capable GPU
GNU Make 4.3
CMake version 3.22 (required to be 3.22 or above)
Boost Library (used 1.81.0) (save it in the folder: /usr/include/)

IMPORTANT NOTES:
- To initialize the project, enter the "SOURCE_FILES" folder, and type "make " with the specific option you want:
    - "make compile": erase the previous build and SIMULATION_OUTPUT folders and compiles the solution.
    - "make total": Same as before, but runs the executable afterwards.
    - "make update": Updates files in the build directory and executes the executable.
  We recommend to use "Compile" and "Total" on the first time it is executed in a computer, and use "Update" afterwards.
  The executable is generated in "/SOURCE_FILES/build".

- To run the simulation (without the make commands), go to "SOURCE_FILES/build/" and then call the executable "./Simulation"

- Note that when you recompile, the "SIMULATION OUTPUT" folder is cleaned. Save your outputs somewhere else, or deactivate this feature
  in the makefile document.

- There are several examples in 1, 2 and even 3 dimensions with several combinations of parameters, impulse terms, etc. I believe it is
  well-explained, but feel free to experiment with different combinations. Note that different combinations may imply that time-stepping,
  number of parameter samples or even domain size must be changed.

- DO NOT CHANGE NAMES OF VARIABLES.

- The simulator will adapt to your GPU memory availability as long as you have the amount of memory necessary to hold the corresponding number
  of variables for 1 parameter sample (that is, if there is memory for 1 parameter sample, there is memory for 400, for example).

- Note that the variable used for storage will grow very fast when performing simulations in >2D for a large amount of timesteps.

- We have added the possibility of storing full, or partial information in separate '.csv' files, where each file can be 1 GB at most. This
  can also be changed in the 'Case_definition.cuh' file. 

- There are Matlab scripts to visualize the information obtained from the simulation (currently it is separated for 1, 2 and 3D cases).
  Feel free to experiment with the visualization tools.

For any other doubts, feel free to ask for help at: vizenzo97@gmail.com or vibees@doctor.upv.es (until December)