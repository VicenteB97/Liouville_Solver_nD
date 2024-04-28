# Liouville Solver

In this repository we have a CUDA solution for the computation of the solution to the Liouville Equation in general dimensions.
We will explain the steps so you can familiarize with the code and test it for yourself! (Any optimizations/errors, please let me know)

## REQUIREMENTS

- LINUX/WSL:

  - g++ compiler (g++ >= 11.3.0)
  - nvcc compiler (nvcc >= 11.5) -- Obviously you need a CUDA-Capable GPU
  - GNU Make >= 4.3
  - CMake >= 3.22 (required)
  - Boost Library (used 1.81.0) (save it in the folder: /usr/include/)

- WINDOWS:
  - C++ compiler: MSVC (>= 19.36)
  - CUDA compiler: NVIDIA >= 12.2.91
  - CMake >= 3.22 (required)
  - Boost Library (save the include folder in the include folder of MSVC)

## IMPORTANT NOTES

- Set the build system in the CMakeLists.txt file (Linux or Windows)
- To initialize the project:
  - **LINUX** Type "make " with the specific option you want:
    - "make compile_X": erase the previous build and SIMULATION_OUTPUT folders and compiles the solution.
    - "make total_X": Same as before, but runs the executable afterwards.
    - "make update_X": Updates files in the build directory and executes the executable.
      - Use "Compile" and "Total" on the first time it is executed in a computer, and use "Update" afterwards.
      - The executable is generated in "./build/X".
      - Options for X are 'debug' and 'release'.

  - **WINDOWS** Type cmake -S . -B ./build
  After it's done, type cmake --build ./build --config Y (Y can be either 'Debug' or 'Release', without the quotes)

- To run the simulation (without the make commands), go to "./build/app/Y" and then call the executable:
  - "./Simulation" for Linux/WSL
  - "start Simulation.exe" for Windows

- There are several examples in 1, 2 and even 3 dimensions with several combinations of parameters, impulse terms, etc. I believe it is
  well-explained, but feel free to experiment with different combinations. Note that different combinations may imply that time-stepping,
  number of parameter samples or even domain size must be changed.

- ***DO NOT CHANGE NAMES OF VARIABLES*** in the 'Case_definition.cuh' document.

- The simulator will adapt to your GPU memory availability as long as you have the amount of memory necessary to hold the corresponding number
  of variables for 1 parameter sample (that is, if there is memory for 1 parameter sample, then you can run for 400, for example).

- Note that the variable used for storage will grow very fast when performing simulations in >2D for a large amount of timesteps.

- Added the possibility of storing full, or partial information in separate '.bin' (binary) files, where each file can be 4 GB at most. This can also be changed in the 'Case_definition.cuh' file. Files are stored in parallel, but the naming makes sure that the simulation is stored correctly.

- There are Matlab scripts to visualize the information obtained from the simulation (currently it is separated for 1, 2 and 3D cases). Feel free to experiment with the visualization tools.

For any other doubts, feel free to ask for help at: <vizenzo97@gmail.com> or <vibees@upv.es>
