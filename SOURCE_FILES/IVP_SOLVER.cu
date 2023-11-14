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
int16_t PDF_EVOLUTION(cudaDeviceProp* prop) {
	std::cout << "You are simulating the " << CASE <<". Log level is (0 = off, 1 = only frame time, 2 = all steps time): " << OUTPUT_INFO << ".\n\n";
	std::cout << "You must choose: \n - FINEST MESH LEVEL \n - FINAL time \n - TIMESTEP";
	std::cout << "\n - Reinitialization Steps \n - SAMPLES PER PARAMETER \n";
	std::cout << "(REMINDER) You can type \"-1\" to  exit the program while typing this information.\n\n";
	// Pre-Simulation Data initialization -------------------------------------------------------------------------------

	ivpSolver::ivpSolver<TYPE> Solver;

	Solver.buildDomain();
	Solver.buildTimeVec();
	Solver.getDistributions();

auto start = std::chrono::high_resolution_clock::now();
	
	// error_check = PDF_ITERATIONS<PHASE_SPACE_DIMENSIONS, TYPE>(prop, &storePDF_Frames, Parameter_Mesh, Problem_Domain, Supp_BBox, &H_PDF, n_samples, time_vector, deltaT, ReinitSteps);
	// if (error_check == -1){	std::cout << "An error has occured. Exiting simulation.\n"; return error_check;}

	// int16_t error_check = PDF_ITERATIONS<PHASE_SPACE_DIMENSIONS, PARAM_SPACE_DIMENSIONS, TYPE>(prop, &storePDF_Frames, Problem_Domain, *(&Param_dist), *(&IC_dist_params), time_vector, deltaT, ReinitSteps);
	// if (error_check == -1){	std::cout << "An error has occured. Exiting simulation.\n"; return error_check;}

	int16_t error_check = Solver.EvolvePDF(prop);

auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start; // duration

// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// ---------------------- STORAGE INTO COMPUTER HDD for post-processing ----------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //

	Solver.WriteFramesToFile(duration.count());

	gpuError_Check(cudaDeviceReset());

	return error_check;
}