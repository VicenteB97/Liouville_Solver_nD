#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

// C++/C libraries used
#include <fstream>      // Files output
#include <iostream>     // Command Line input/output
#include <vector>       // vector class is widely used
#include <cmath>        // C library (used for mathematical stuff)
#include <numeric>      // OPTIONAL?
#include <algorithm>    // OPTIONAL?
#include <cstdint>      // C library. Allows the use of specific int and unsigned int types, such as uint16_t
#include <string>       // For strings
#include <omp.h>        // For simple use of thread-parallel computations
#include <chrono>       // Timing the execution of functions
#include <limits>		// To get limits from variables

// Headers for the CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Thrust libraries
#include <thrust/host_vector.h>         // To use the host vector class. Equivalent to std::vector
#include <thrust/device_vector.h>       // GPU-equivalent of std::vector
#include <thrust/reduce.h>              // To compute the parallel reduction of vectors
#include <thrust/execution_policy.h>    // Important to specify where the reduction is taking place
#include <thrust/functional.h>          // OPTIONAL?
#include <thrust/transform.h>           // To multiply vectors using a parallel reduction-like fashion
#include <thrust/fill.h>                // Explicitly fill host/device functions with a value
#include <thrust/sort.h>                // GPU-based sorting
#include <thrust/extrema.h>             // Compute max. and min. of GPU arrays

// Include the Simulation definition!
#include "Case_definition.cuh"			// Here, we read the information of the problem to be simulated!
#include "config.hpp"

// Simulation and logging default output folder
#define SIM_OUTPUT_relPATH "../../SIMULATION_OUTPUT/"
#define LOG_OUTPUT_relPATH "../../SIMULATION_OUTPUT/"

// Compute + storage type
#define TYPE float

// Maximum simulation file size
#define MAX_FILE_SIZE_B (uint64_t) 4*1024*1024*1024

// Check the dimensionality of the simulations to use "optimal" data types
#if PHASE_SPACE_DIM < 4
	#define INT int32_t
	#define UINT uint32_t
#else
	#define INT int64_t
	#define UINT uint64_t
#endif

// Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// Mathematical definitions we need
#define M_PI 3.14159265358979323846
#define ptSEARCH_THRESHOLD 1000000 // The min. number of particles per sample where we use the counting sort search (not worth it yet)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// GPU-version of the Assert function
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort = true){
if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Some more defines that shouldn't be modified
#define gpuError_Check(ans) {gpuAssert((cudaError_t) ans, __FILE__, __LINE__);}

// Error checking in the CPU code
#define errorCheck(ans) {if(ans == -1){std::cout << "Error found at:\n" << __FILE__ << "\nLine: " << std::to_string(__LINE__) << ".\n"; return -1;}}

// This is for the thrust library
#define rpc(ans,offset) raw_pointer_cast(&ans[offset])

// This function is widely used throughout the code so...quite effective
/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a This is the numerator
/// @param b This is the denominator
/// @return Returns mod(a,b)
__host__ __device__
inline UINT positive_rem(const INT a, const INT b) {
	return (a % b + b) % b;
}

/// @brief 
/// @param inputTerminal 
/// @return 
inline bool isNumeric(const std::string& inputTerminal){

	// Iterate through every character in the string and check if they're numbers
	int16_t strLength = inputTerminal.size(), CharPosition = 0;
	bool isDecimal = false;

	// It's OK if the first term is a minus sign
	if(inputTerminal[0] == '-'){
		CharPosition++;
	}
	// It's OK if the first term is a dot character
	else if(inputTerminal[0] == '.'){
		CharPosition++;
		isDecimal = true;
	}

	// Go through all characters in the string
	while(CharPosition < strLength){

		// Check if the character is a digit
		if(!std::isdigit(inputTerminal[CharPosition])){
			
			// Check if it's the dot character and wether it has already appeared
			if(inputTerminal[CharPosition] == '.' && !isDecimal){
				isDecimal = true;
				CharPosition++;
			}
			else{
				return false;
			}
		}
		CharPosition++;
	}
	return true;

}

#define DISTR_ERR_MSG "At least 1 sample must be chosen. Try again...\n"
#define DOMAIN_ERR_MSG "You must choose a STRICTLY positive integer!. Try again...\n"
#define INIT_TIME_ERR_MSG "You must choose a STRICTLY positive initial time.\n"
#define END_TIME_ERR_MSG "You must choose a STRICTLY positive end time. Also, has to be greater than init. time!\n"
#define TIMESTEP_ERR_MSG "You must choose a STRICTLY positive timestep.\n"
#define REINIT_ERR_MSG "You must choose a STRICTLY positive number of steps.\n"

inline int16_t intCheck(bool& getAnswer, const std::string& inputTerminal, const std::string& errMessage = "Undefined error occured.\n", const INT non_accepted = 0, const INT minArg = std::numeric_limits<INT>::lowest(), 
						const INT maxArg = std::numeric_limits<INT>::max()) {
	
	if (!isNumeric(inputTerminal)) { std::cout << "Error: Non-numerical inputs not allowed. "; }
	else {

		INT temp = std::stoi(inputTerminal);

		if (temp == -1) {
			std::cout << "Definition error in file: " << __FILE__ << "\nLine: " << __LINE__ << "\nExiting simulation.\n";
			return -1;
		}

		if (temp < minArg || temp == non_accepted || temp > maxArg) {
			std::cout << errMessage;
		}
		else {
			getAnswer = false;
		}
	}

	return 0;
	
}

inline int16_t doubleCheck(bool& getAnswer, const std::string& inputTerminal, const std::string& errMessage = "Undefined error occured.\n", const double non_accepted = 0, const double minArg = std::numeric_limits<double>::lowest(),
							const double maxArg = std::numeric_limits<double>::max()) {

	if (!isNumeric(inputTerminal)) { std::cout << "Error: Non-numerical inputs not allowed. "; }
	else {

		double temp = std::stod(inputTerminal);

		if (temp == -1) {
			std::cout << "Definition error in file: " << __FILE__ << "\nLine: " << __LINE__ << "\nExiting simulation.\n";
			return -1;
		}

		if (temp == non_accepted || temp < minArg || temp > maxArg) {
			std::cout << errMessage;
		}
		else {
			getAnswer = false;
		}
	}

	return 0;
	
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STUFF RELATED TO THE OUTPUT TERMINAL WINDOW
#define border "+==========================================================================================+"
#define border_mid "|==========================================================================================|"
#define windowLength 92

void Intro_square_filler(std::string& message, const uint32_t border_length, const uint32_t number_of_tabs, const uint32_t offset){

	uint32_t msg_length = message.size() + number_of_tabs*8 - 1 - offset;
	for (uint32_t k = 0; k < border_length - msg_length; k++){
		message.append(" ");
	}
	message.append("|\n");
}

int16_t SetGPU(int16_t& ChosenOne, const uint32_t& border_length){

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		printf("No CUDA-supported GPU found. Exiting program.\n");
		std::cin.get();
		return -1;
	}

	ChosenOne = 0;

	// If we have more than one, we choose the one with most memory
	std::string temp = "|	Found " ;
		temp.append(std::to_string(deviceCount));
		temp.append(" GPUs.");
		Intro_square_filler(temp, border_length, 1, 0);
		std::cout << temp;
    	
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, ChosenOne);

		uint64_t max_memory = properties.totalGlobalMem;

		for (uint16_t device = 0; device < deviceCount; device++) {

			cudaGetDeviceProperties(&properties, device);
			uint64_t temp_mem = properties.totalGlobalMem;

			temp =  "|		- GPU ";
			temp.append(std::to_string(device));
			temp.append(" is: ");
			temp.append(properties.name);
			temp.append(" with ");
			temp.append(std::to_string(temp_mem/1024/1024/1024));
			temp.append(" GB of global memory.");
			Intro_square_filler(temp, border_length, 2, 1);
			std::cout << temp;

			if (max_memory < temp_mem) {
				max_memory = temp_mem;
				ChosenOne = device;
			}
		}
	return 0;
}

int16_t IntroDisplay(cudaDeviceProp& prop){
	std::cout << border <<"\n";
	std::cout << border_mid <<"\n";
	const uint32_t border_length = windowLength;

	std::string temp = "|	Wecome to the Liouville Eq. simulator. You are using version number "; temp.append(project_version);
	Intro_square_filler(temp, border_length, 1, 0);
	std::cout << temp;
	std::cout << border_mid <<"\n";

	// Choose GPU to use
	int16_t chosen_GPU, 
			err_code = SetGPU(chosen_GPU, border_length);
	if( err_code == -1 ){ return err_code; }
	
	cudaGetDeviceProperties(&prop, chosen_GPU);

	temp = "|	Starting simulation using GPU "; temp.append(std::to_string(chosen_GPU)); temp.append(". Properties:");
	Intro_square_filler(temp, border_length, 1, 0);
	std::cout << temp;

	temp = "|		- Global memory available (GB): "; temp.append(std::to_string(prop.totalGlobalMem/1024/1024/1024));
	Intro_square_filler(temp, border_length, 2, 1);
	std::cout << temp;

	temp = "|		- Max. memory bus width: "; temp.append(std::to_string(prop.memoryBusWidth)); temp.append(" bits.");
	Intro_square_filler(temp, border_length, 2, 1);
	std::cout << temp;

	temp = "|		- Number of asynchronous engines: " ; temp.append(std::to_string(prop.asyncEngineCount));
	Intro_square_filler(temp, border_length, 2, 1);
	std::cout << temp;

	temp = "|		- Number of SMs: "; temp.append(std::to_string(prop.multiProcessorCount));
	Intro_square_filler(temp, border_length, 2, 1);
	std::cout << temp;

	temp = "|		- Max. threads per SM: "; temp.append(std::to_string(prop.maxThreadsPerMultiProcessor));
	Intro_square_filler(temp, border_length, 2, 1);
	std::cout << temp;

	temp = "|		- Four-byte registers per SM: "; temp.append(std::to_string(prop.regsPerMultiprocessor));
	Intro_square_filler(temp, border_length, 2, 1);
	std::cout << temp;

	std::cout << border_mid << "\n";
	std::cout << border << "\n";

	return 0;
}

#endif