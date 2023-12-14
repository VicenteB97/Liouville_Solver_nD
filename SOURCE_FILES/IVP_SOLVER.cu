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

#include "ivpSolver.cuh"

// --------------------------------------------------------- //
// --------------------------------------------------------- //

//--------------------------------------------------------------------------------------------- //
int16_t PDF_EVOLUTION() {

	cudaDeviceProp prop;
	errorCheck(IntroDisplay(prop));

	std::cout << "You are simulating the " << CASE <<". Log level is (0 = off, 1 = only frame time, 2 = all steps time): " << OUTPUT_INFO << ".\n\n";
	std::cout << "You must choose: \n - FINEST MESH LEVEL \n - FINAL time \n - TIMESTEP";
	std::cout << "\n - Reinitialization Steps \n - SAMPLES PER PARAMETER \n";
	std::cout << "(REMINDER) You can type \"-1\" to  exit the program while typing this information.\n\n";
	// Pre-Simulation Data initialization -------------------------------------------------------------------------------

	ivpSolver::ivpSolver Solver;

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
	gpuError_Check(cudaDeviceReset());

	return 0;
}