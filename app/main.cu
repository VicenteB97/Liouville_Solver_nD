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
#include "terminal/terminal.hpp"

int16_t PDF_EVOLUTION();

int main() {

	printEntryMessage();
	return PDF_EVOLUTION();
}

//--------------------------------------------------------------------------------------------- //
int16_t PDF_EVOLUTION() {

	ivpSolver Solver;

	errorCheck(Solver.buildDomain())
	
	errorCheck(Solver.buildTimeVec())

	errorCheck(Solver.buildDistributions())

auto start = std::chrono::high_resolution_clock::now();

	errorCheck(Solver.evolvePDF())

auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start; // duration

// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// ---------------------- STORAGE INTO COMPUTER HDD for post-processing ----------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //

	errorCheck(Solver.writeFramesToFile(duration.count()));

	return 0;
}