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
#include <string>

#include "ivpSolver/ivpSolver.hpp"
#include "include/utils/window.cuh"

int16_t PDF_EVOLUTION();

int main() {
	//----------------------------------------------------------------------------------------------------------------------------------------//

	int16_t ret_val = PDF_EVOLUTION();

	std::string temp = "|	Simulation finished successfully.\n";
	Intro_square_filler(temp, windowLength, 1, 0);
	std::cout << border_mid << "\n" << border << "\n";
	
	return ret_val;
}

//--------------------------------------------------------------------------------------------- //
int16_t PDF_EVOLUTION() {

	cudaDeviceProp prop;
	errorCheck(IntroDisplay(prop));

	std::string temp = "|	You are simulating " + (std::string)CASE;
	Intro_square_filler(temp, windowLength, 1, 0);
	std::cout << border_mid << "\n";
	// Pre-Simulation Data initialization -------------------------------------------------------------------------------

	ivpSolver Solver;

	errorCheck(Solver.buildDomain())
	
	errorCheck(Solver.buildTimeVec())

	errorCheck(Solver.buildDistributions())

auto start = std::chrono::high_resolution_clock::now();

	errorCheck(Solver.evolvePDF(prop))

auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start; // duration

// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// ---------------------- STORAGE INTO COMPUTER HDD for post-processing ----------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //

	errorCheck(Solver.writeFramesToFile(duration.count()))
	errorCheck_dvc(cudaDeviceReset());

	return 0;
}