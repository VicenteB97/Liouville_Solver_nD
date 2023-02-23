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

The code was used to perform the computations in the numerical simulations in the following contributions:
/ INCLUDE PAPERS WHEN IT IS PUBLISHED /

INSTRUCTIONS TO REPLICATE EXPERIMENTS IN ARTICLE 1:
1.- In "Classes.cuh", you must select either "#define CASE 1" for the 1D example, or "#define CASE 2" for the 2D example.
2.- Next, go to the directory where the downloaded files are located. In the terminal, write "make total".
3.- When the welcome screen appears, you have to select the following parameters for each case (press Enter after writing each parameter):
  3.1 - 10, 0, 45, 0.25, 4, 20, 20
  3.2 - 9, 0, 2.6, 0.05, 2, 7, 15
4.- Press Enter and let the simulation finish
5.- Press "Y" or "N" to save, or not, the simulation output. These .csv files will be stored inside the "build" directory built by CMake.
6.- Use the "READ_CPP_1D.m" or "READ_CPP.m" files in Matlab/Octave to visualize the simulation output.

Note that timing and memory limits are device-dependent. We advise to have, at least, 3 GB of VRAM available when performing the simulations.
We are currently working in a way to dramatically reduce the memory usage.