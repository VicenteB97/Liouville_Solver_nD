#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "config.hpp"

int PDF_EVOLUTION();

int main() {
	// This file is going to have all the information regarding the steps to follow for solving the Liouville Equation associated to a RDE System

	// DYNAMICS is going to have all the information about the vector field defined by the RDE System
	// PARAMETER_DISTRIBUTIONS contains the definition of the PDFs of all the random/deterministic parameters involved in the RDE System
	// ADAPT_MESH contains all that is related to the Adaptive Mesh Refinement (AMR) Procedure
	// RANGE_SEARCH contains the definitions used for the SRS procedure
	// MATRIX_VECTOR defines the special Matrix-vector multiplication scheme used in these computations

	// FULL_SIMULATION is where everything is combined to compute the evolution of the initial PDF through time via the Liouville Equation

	// Note: All the compuations are saved in a .xls document for post_processing in Matlab/Python

	//----------------------------------------------------------------------------------------------------------------------------------------//
	// STEPS:
	// 1.- Define the phase space DIMENSIONS and your vector field (DYNAMICS)
	// 2.- Define the parameter distributions (PARAMETER_DISTRIBUTIONS)
	// 3.- Define the simulation parameters (SIMULATION_PARAMETERS)
	// 4.- Change number of samples and RANDOMIZE function at the FULL_SIMULATION.cuh
	// 
	// (OPTIONAL) 5.- Change the translation vector at RANGE_SEARCH.cuh if there is delta-type impulse in your dynamics
	//----------------------------------------------------------------------------------------------------------------------------------------//
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("No CUDA-supported GPU found. Exiting program.\n");
		std::cin.get();
		return -1;
	}
	else {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		std::cout << "// --------------------------------------------------------------------------------------- //\n";
		std::cout << "	 Wecome to the Liouville Eq. simulator. You are using version number " << project_version << "\n";

		printf("	 Starting simulation using device: %s  \n", prop.name);
		std::cout << "// --------------------------------------------------------------------------------------- //\n";
		return PDF_EVOLUTION();
	}
}