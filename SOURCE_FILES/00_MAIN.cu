//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//														//
//		Written by: Vicente José Bevia Escrig			//
//		Mathematics Ph.D. student (2020-2024) at:		//
//		Instituto de Matemática Multidisciplinar,		//
//		Universitat Politècnica de València, Spain		//
//														//
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////


#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "config.hpp"
#include "Case_definition.cuh"

int PDF_EVOLUTION(cudaDeviceProp* prop);

void Intro_square_filler(std::string& message, const u_int16_t border_length, const u_int16_t number_of_tabs, const u_int16_t offset){

	u_int16_t msg_length = message.size() + number_of_tabs*8 - 1 - offset;
	for (u_int16_t k = 0; k < border_length - msg_length; k++){
		message.append(" ");
	}
	message.append("|\n");
}

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

		std::string border 		= "+==========================================================================================+";
		std::string border_mid 	= "|==========================================================================================|";
		std::cout << border <<"\n";
		std::cout << border_mid <<"\n";
		const u_int16_t border_length = border.size();

		std::string temp = "|	Wecome to the Liouville Eq. simulator. You are using version number "; temp.append(project_version);
		Intro_square_filler(temp, border_length, 1, 0);
		std::cout << temp;
		std::cout << border_mid <<"\n";

		temp = "|	Starting simulation using device: "; temp.append(prop.name); temp.append(". Properties:");
		Intro_square_filler(temp, border_length, 1, 0);
		std::cout << temp;

		temp = "|		- Global memory available (GB): "; temp.append(std::to_string(prop.totalGlobalMem/1024/1024/1024));
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

	#if OUTPUT_INFO
		temp = "|		- Max. memory bus width: "; temp.append(std::to_string(prop.memoryBusWidth)); temp.append(" bits.");
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

		temp = "|		- Number of SMs: "; temp.append(std::to_string(prop.multiProcessorCount));
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

		temp = "|		- Number of asynchronous engines: " ; temp.append(std::to_string(prop.asyncEngineCount));
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

		temp = "|		- Max. threads per SM: "; temp.append(std::to_string(prop.maxThreadsPerMultiProcessor));
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

		temp = "|		- Number of SMs: "; temp.append(std::to_string(prop.multiProcessorCount));
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

		temp = "|		- Four-byte registers per SM: "; temp.append(std::to_string(prop.regsPerMultiprocessor));
		Intro_square_filler(temp, border_length, 2, 1);
		std::cout << temp;

	#endif

		std::cout << border_mid << "\n";
		std::cout << border << "\n";

		return PDF_EVOLUTION(&prop);
	}
}