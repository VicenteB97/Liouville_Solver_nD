#pragma once

#include "../headers.cuh"
#include "numeric_defs.cuh"
#include "../../build/configured_files/include/config.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// STUFF RELATED TO THE OUTPUT TERMINAL WINDOW
#define border "+==========================================================================================+"
#define border_mid "|==========================================================================================|"
#define windowLength 92

inline void Intro_square_filler(std::string& message, const uint32_t border_length, const uint32_t number_of_tabs, const uint32_t offset){

	uint32_t msg_length = message.size() + number_of_tabs*8 - 1 - offset;
	for (uint32_t k = 0; k < border_length - msg_length; k++){
		message.append(" ");
	}
	message.append("|\n");
}

inline int16_t SetGPU(int16_t& ChosenOne, const uint32_t& border_length){

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

inline int16_t IntroDisplay(cudaDeviceProp& prop){
	std::cout << border <<"\n";
	std::cout << border_mid <<"\n";
	const uint32_t border_length = windowLength;
	const std::string str_project_version{project_version};

	std::string temp = "|	Wecome to the Liouville Eq. simulator. You are using version number "; temp += str_project_version;
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