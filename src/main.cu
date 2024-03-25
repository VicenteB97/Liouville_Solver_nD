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

#include "../include/Case_definition.cuh"
#include <string>

int16_t PDF_EVOLUTION();

int main() {
	//----------------------------------------------------------------------------------------------------------------------------------------//

	int16_t ret_val = PDF_EVOLUTION();

	std::cout << "Simulation finished. Press any key to continue...\n";
	std::cin.ignore();
	std::cin.get();

	return ret_val;
}