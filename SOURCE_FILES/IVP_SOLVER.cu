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

#include "Liouville_iteration.cuh"

// --------------------------------------------------------- //
// --------------------------------------------------------- //

//--------------------------------------------------------------------------------------------- //
int16_t PDF_EVOLUTION(cudaDeviceProp* prop) {
	std::cout << "You are simulating the " << CASE <<". Log level is (0 = off, 1 = only frame time, 2 = all steps time): " << OUTPUT_INFO << ".\n\n";
	std::cout << "You must choose: \n - FINEST MESH LEVEL \n - FINAL time \n - TIMESTEP";
	std::cout << "\n - Reinitialization Steps \n - SAMPLES PER PARAMETER \n";
	std::cout << "(REMINDER) You can type \"-1\" to  exit the program while typing this information.\n\n";
	// Pre-Simulation Data initialization -------------------------------------------------------------------------------

	// ----------------------------------------------------------------------------------------------- //
	// ---------------------------------- OBTAIN INFO FROM TERMINAL ---------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	INT LvlFine;
	std::cout << "Finest level in the domain?: ";
	std::cin >> LvlFine;
	if (LvlFine == -1){
		std::cout << "Exiting simulation\n";
		return -1;
	}
	
	// This variable represents the problem domain, which is NOT going to be the one used for computations
	grid<DIMENSIONS, TYPE> Problem_Domain((INT) pow(2, LvlFine));

	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //

// -------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------- //
	// Time simulation data Definition: -------------------------------------------------------------------------------
	INT	ReinitSteps;
	double	deltaT;

	std::vector<Time_Impulse_vec> time_vector;
	
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	int16_t error_check = 0;

	error_check = Simul_Data_Def(time_vector, deltaT, ReinitSteps);
	if (error_check == -1){	std::cout << "Exiting simulation.\n"; return -1;}

	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// ---------------------- PDF EVOLUTION STEPS AND SIMULATION RUN ------------------------------
	// --------------------------------------------------------------------------------------------
	// 	   Note: If we want to do some kind of optimzation procedure regarding the Liouville Eq.,
	// 	   you have to make a function such as PDF_EVOLUTION but you must modify THIS following part
	// 	   PDF_EVOLUTION simply computes the evolution of a PDF according to some sim. parameters
	// 	   as well as the corresponding dynamics parameters.
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// 1.- PARAMETERS biuld up
		Distributions 	Param_dist[PARAM_DIMENSIONS];		// Array for storing the model parameters' distribution information

		for (UINT p = 0; p < PARAM_DIMENSIONS; p++){
			Param_dist[p].Name  			= _DIST_NAMES[p];		// N, G or U distributions
			Param_dist[p].Truncated  		= _DIST_TRUNC[p];		// TRUNCATED?
			Param_dist[p].trunc_interval[0] = _DIST_InfTVAL[p];		// min of trunc. interval
			Param_dist[p].trunc_interval[1] = _DIST_SupTVAL[p]; 	// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
			Param_dist[p].params[0] 		= _DIST_MEAN[p];		// mean
			Param_dist[p].params[1] 		= _DIST_STD[p];			// std

			// Read samples from terminal
			bool get_answer = true;
			INT temp = 1;

			while (get_answer) {
				std::cout << "How many samples for parameter " << p + 1 << " ? ";
				std::cin >> temp;

				if (temp == -1) {
					std::cout << "Exiting simulation.\n"; return -1;
				}
				else if (temp == 0) {
					std::cout << "At least 1 sample must be selected. ";
				}
				else {
					get_answer = false;
				} 
			}
			Param_dist[p].num_Samples = temp;
		}

		Distributions IC_dist_params[DIMENSIONS];

		for (UINT d = 0; d < DIMENSIONS; d++) {
			IC_dist_params[d].Name = IC_NAMES[d];
			IC_dist_params[d].Truncated = IC_TRUNC[d];
			IC_dist_params[d].trunc_interval[0] = IC_InfTVAL[d];
			IC_dist_params[d].trunc_interval[1] = IC_SupTVAL[d];
			IC_dist_params[d].params[0] = IC_MEAN[d];
			IC_dist_params[d].params[1] = IC_STD[d];
			IC_dist_params[d].num_Samples = -1;		// This value is just for information. Samples are not chosen here
		}

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// 3.- Evolution simulation

	std::vector<float>	store_PDFs(Problem_Domain.Total_Nodes() * time_vector.size());		 // H_PDF storage for post-processing

auto start = std::chrono::high_resolution_clock::now();
	
	// error_check = PDF_ITERATIONS<DIMENSIONS, TYPE>(prop, &store_PDFs, Parameter_Mesh, Problem_Domain, Supp_BBox, &H_PDF, n_samples, time_vector, deltaT, ReinitSteps);
	// if (error_check == -1){	std::cout << "An error has occured. Exiting simulation.\n"; return error_check;}

	error_check = PDF_ITERATIONS<DIMENSIONS, PARAM_DIMENSIONS, TYPE>(prop, &store_PDFs, Problem_Domain, *(&Param_dist), *(&IC_dist_params), time_vector, deltaT, ReinitSteps);
	if (error_check == -1){	std::cout << "An error has occured. Exiting simulation.\n"; return error_check;}

auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start; // duration

// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// ---------------------- STORAGE INTO COMPUTER HDD for post-processing ----------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------- //

// We have added the capability of automatically detecting the number of 1 GB files where we can store the simulation output
	bool saving_active = true;		// see if saving is still active

	const uint64_t MEM_2_STORE		= store_PDFs.size() * sizeof(float);
	
	UINT number_of_frames_needed 	= MEM_2_STORE / Problem_Domain.Total_Nodes() / sizeof(float);
	uint64_t max_frames_file 		= (uint64_t)MAX_FILE_SIZE_B / Problem_Domain.Total_Nodes() / sizeof(float);
	UINT number_of_files_needed  	= floor((number_of_frames_needed - 1) / max_frames_file) + 1;
	
	char ans;
	std::cout << "Simulation time: " << duration.count() << " seconds. ";
	
	if(number_of_files_needed == 0){
		std::cout << "There has been a problem. No memory written. Exiting simulation.\n";
		saving_active = false;
		error_check = -1;
	}

	while(saving_active){	
		std::cout << "Total memory of simulation: " << (float) MEM_2_STORE / 1024/1024 << " MB. \n";
		std::cout << number_of_files_needed << " files required for total storage. Total frames: " << number_of_frames_needed << ", with frames per file: " << max_frames_file << " \n";
		std::cout << "Write? (Y = Yes(total), N = No, P = Partial): ";
		std::cin >> ans;

		while((ans != 'N') && (ans != 'n') && (ans != 'Y') && (ans != 'y') && (ans != 'P') && (ans != 'p')){
			std::cout << "Incorrect option. Choose one of the following (NOT case sensitive: Y = Yes, N = No, P = Partial): ";
			std::cin >> ans;
		}


		if ((ans != 'N') && (ans != 'n')) {

			INT frames_init = 0, frames_end = number_of_files_needed - 1;
			bool condition = false;

			if((ans == 'P') || (ans == 'p')){
				while(!condition){
					std::cout << "Initial frame (must be >= 0): ";
					std::cin >> frames_init;
					std::cout << "Final frame (must be < "<< number_of_frames_needed <<"): ";
					std::cin >> frames_end;

					if(frames_init < 0 || frames_end >= number_of_frames_needed || frames_init > frames_end){

						if(frames_init == -1 || frames_end == -1){
							std::cout << "Exiting simulation without saving simulation results...\n";
							return -1;
						}
						std::cout << "Check numbers, something's not right...\n";
					}
					else{
						condition = true;
						number_of_frames_needed = frames_end - frames_init + 1;
						number_of_files_needed  = floor((number_of_frames_needed - 1) / max_frames_file) + 1;
					}
				}
			}
			else{
				frames_init = 0, frames_end = number_of_files_needed - 1;
			}

			#pragma omp parallel for
			for(int16_t k = 0; k < number_of_files_needed; k++){

				UINT frames_in_file = fmin(max_frames_file, number_of_frames_needed - k * max_frames_file);

				std::string temp_str = std::to_string((UINT)k);

			// SIMULATION INFORMATION FILE
				std::string relavtive_pth = SIM_OUTPUT_relPATH;
				relavtive_pth.append("Simulation_info_");
				relavtive_pth.append(temp_str);
				relavtive_pth.append(".csv");

				std::ofstream file1(relavtive_pth, std::ios::out);
				assert(file1.is_open());

				file1 << Problem_Domain.Total_Nodes() << "," << Problem_Domain.Nodes_per_Dim << ",";

				for (UINT d = 0; d < DIMENSIONS; d++) {
					file1 << Problem_Domain.Boundary_inf.dim[d] << "," << Problem_Domain.Boundary_sup.dim[d] << ",";
				}
				for (uint16_t d = 0; d < PARAM_DIMENSIONS; d++) {
					file1 << Param_dist[d].num_Samples << ",";
				}
				file1 << duration.count() << "\n";

				#if IMPULSE_TYPE == 0 || IMPULSE_TYPE ==1
				for (UINT i = k * max_frames_file + frames_init; i < k * max_frames_file + frames_in_file + frames_init - 1; i++) {
					file1 << time_vector[i].time << ",";
				}
				file1 << time_vector[k * max_frames_file + frames_in_file + frames_init - 1].time;

				#elif IMPULSE_TYPE == 2
				file1 << time_vector[k * max_frames_file + frames_init].time << ",";

				for (UINT i = k * max_frames_file + 1 + frames_init; i < k * max_frames_file + frames_init + frames_in_file; i++) {
					if (abs(time_vector[i].time - time_vector[i - 1].time) > pow(10, -7)) {
						file1 << time_vector[i].time << ",";
					}
					else if (i == k * max_frames_file + frames_in_file + frames_init - 1) {
						if (time_vector[i].time != time_vector[i - 1].time) {
							file1 << time_vector[i].time;
						}
					}
				}
				#endif
				file1.close();

			// SIMULATION OUTPUT
				relavtive_pth = SIM_OUTPUT_relPATH;
				relavtive_pth.append("Mean_PDFs_");
				relavtive_pth.append(temp_str);
				relavtive_pth.append(".bin");

				std::ofstream myfile(relavtive_pth, std::ios::out | std::ios::binary);
				assert (myfile.is_open());

				myfile.write((char*)&store_PDFs[(k * max_frames_file + frames_init) * Problem_Domain.Total_Nodes()], sizeof(float) * frames_in_file * Problem_Domain.Total_Nodes());
				myfile.close();
				std::cout << "Simulation output file " << k << " completed!\n";
				
			}
		}

		saving_active = false;
	}

	gpuError_Check(cudaDeviceReset());

	return error_check;
}