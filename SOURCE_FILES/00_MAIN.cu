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
#include <string>

int16_t PDF_EVOLUTION(cudaDeviceProp* prop);

void Intro_square_filler(std::string& message, const uint16_t border_length, const uint16_t number_of_tabs, const uint16_t offset){

	uint16_t msg_length = message.size() + number_of_tabs*8 - 1 - offset;
	for (uint16_t k = 0; k < border_length - msg_length; k++){
		message.append(" ");
	}
	message.append("|\n");
}

int main() {
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
		const uint16_t border_length = border.size();

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

	#endif

		std::cout << border_mid << "\n";
		std::cout << border << "\n";

		int16_t ret_val = PDF_EVOLUTION(&prop);

		std::cout << "Simulation finished with output code " << ret_val << std::endl;

		std::cout << "Press any key to exit simulation program...";
		std::cin.ignore();	// this part is done so that it ignores a mysterious endline character that appears after the previous line
		std::cin.get();

		return ret_val;
	}
}