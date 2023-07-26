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

#include "Dynamics.cuh"

// --------------------------------------------------------- //
// --------------------------------------------------------- //

//--------------------------------------------------------------------------------------------- //
int16_t PDF_EVOLUTION(cudaDeviceProp* prop) {
	std::cout << "You are simulating the " << CASE <<". Simulation timing information is (1 = on, 0 = off): " << OUTPUT_INFO << ".\n\n";
	std::cout << "You must choose: \n - FINEST / COARSEST MESH LEVEL \n - FINAL time \n - TIMESTEP";
	std::cout << "\n - Reinitialization Steps \n - SAMPLES PER PARAMETER \n";
	std::cout << "(REMINDER) You can type \"-1\" to  exit the program while typing this information.\n\n";
	// Pre-Simulation Data initialization -------------------------------------------------------------------------------

	// ----------------------------------------------------------------------------------------------- //
	// ---------------------------------- OBTAIN INFO FROM TERMINAL ---------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	int32_t LvlFine, LvlCoarse = 0;
	std::cout << "Finest level in the domain?: ";
	std::cin >> LvlFine;
	if (LvlFine == -1){
		std::cout << "Exiting simulation\n";
		return -1;
	}

	std::cout << "Coarsest level in the domain?: ";
	std::cin >> LvlCoarse;
	if (LvlCoarse == -1){
		std::cout << "Exiting simulation\n";
		return -1;
	}
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //

	const gridPoint Domain_Center 	= DOMAIN_CTR;
	const gridPoint Domain_Diameter = DOMAIN_DIAM;

	const uint32_t PtsPerDim  	= pow(2, LvlFine);
	const uint32_t Grid_Nodes 	= pow(PtsPerDim, DIMENSIONS);
	gridPoint* H_Mesh 	 		= new gridPoint[Grid_Nodes];

	// GENERAL DIMENSION Cartesian coordinate grid build up
	#pragma omp parallel for
	for (int32_t i = 0; i < Grid_Nodes; i++){
		for (uint32_t d = 0; d < DIMENSIONS; d++){
			uint32_t j 	 = floor(positive_rem(i, pow(PtsPerDim, d + 1))/pow(PtsPerDim, d));
			H_Mesh[i].dim[d] = ((TYPE) j / (PtsPerDim - 1) - 0.50f) * Domain_Diameter.dim[d] + Domain_Center.dim[d]; 
		}
	}

// -------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------- //
	// Time simulation data Definition: -------------------------------------------------------------------------------
	int32_t	ReinitSteps;
	FIXED_TYPE	deltaT;

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
		int32_t n_samples[PARAM_DIMENSIONS];	// number of samples per parameter

		TYPE IC_dist_params[DIMENSIONS * 2];	// distribution parameters for the IC
		
	// ----------------------------------------------------------------------------------------------- //
	// ---------------------------------- OBTAIN INFO FROM TERMINAL ---------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
		for (uint32_t k = 0; k < PARAM_DIMENSIONS; k++) {
			std::cout << "How many samples for parameter " << k + 1 << " ? ";
			std::cin >> n_samples[k];
			while (n_samples[k] == 0){ 
				std::cout << "At least 1 sample must be selected. How many samples for parameter " << k + 1 << " ? "; 
				std::cin >> n_samples[k];
			}
			if (n_samples[k] == -1){
			std::cout << "Exiting simulation.\n"; return -1;}
		}
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
		uint32_t PM_length = 1;															// Total number of parameter samples we use
		for (uint32_t i = 0; i < PARAM_DIMENSIONS; i++){
			PM_length += n_samples[i];
		}

		Param_pair*		Parameter_Mesh 	= new Param_pair[PM_length];				// Full parameter array
		Distributions* 	Param_dist  	= new Distributions[PARAM_DIMENSIONS];		// Array for storing the model parameters' distribution information
	
		thrust::host_vector<TYPE> H_PDF(Grid_Nodes);	 							// PDF values at the fixed, high-res grid (CPU)

		#pragma unroll
		for (uint32_t p = 0; p < PARAM_DIMENSIONS; p++){
			Param_dist[p].Name  			= _DIST_NAMES[p];		// N, G or U distributions
			Param_dist[p].Truncated  		= _DIST_TRUNC[p];		// TRUNCATED?
			Param_dist[p].trunc_interval[0] = _DIST_InfTVAL[p];		// min of trunc. interval
			Param_dist[p].trunc_interval[1] = _DIST_SupTVAL[p]; 	// max. of trunc. interval (if chosen large enough, automatically bounds to 6 std. deviations)
			Param_dist[p].params[0] 		= _DIST_MEAN[p];		// mean
			Param_dist[p].params[1] 		= _DIST_STD[p];			// std
		}

		#pragma unroll
		for (uint32_t d = 0; d < DIMENSIONS; d++){
			IC_dist_params[2*d] 	= IC_MEAN[d];
			IC_dist_params[2*d + 1] = IC_STD[d];
		}

// CALL JOINT PDF BUILDING FUNCTIONS: (PARAMETERS AND INITIAL CONDITION)
	error_check = RANDOMIZE(n_samples, Parameter_Mesh, Param_dist);
	if (error_check == -1){return -1;}

	error_check = PDF_INITIAL_CONDITION(PtsPerDim, H_Mesh, H_PDF, IC_dist_params); 	// initialize the grid and the PDF at the grid nodes (change so as to change the parameters as well)
	if (error_check == -1){return -1;}

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// 3.- Evolution simulation

	std::vector<float>	store_PDFs(Grid_Nodes * time_vector.size());		 // H_PDF storage for post-processing

auto start = std::chrono::high_resolution_clock::now();
	
	error_check = PDF_ITERATIONS(prop, &store_PDFs, Parameter_Mesh, H_Mesh, &H_PDF, n_samples, LvlFine, LvlCoarse, PtsPerDim, Grid_Nodes, time_vector, deltaT, ReinitSteps);
	if (error_check == -1){	std::cout << "An error has occured. Exiting simulation.\n"; return error_check;}

auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start; // duration

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// ---------------------- STORAGE INTO COMPUTER HDD for post-processing -----------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

// We have added the capability of automatically detecting the number of 1 GB files where we can store the simulation output

	bool saving_active = true;		// see if saving is still active

	const uint64_t MEM_2_STORE = store_PDFs.size() * sizeof(float);
	
	uint32_t number_of_frames_needed 	= MEM_2_STORE / Grid_Nodes / sizeof(float);
	uint32_t max_frames_file 			= MAX_FILE_SIZE_B / Grid_Nodes / sizeof(float);
	uint16_t number_of_files_needed  	= floor((number_of_frames_needed - 1) / max_frames_file) + 1;
	
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

			int32_t frames_init = 0, frames_end = number_of_files_needed - 1;
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

				uint16_t frames_in_file = fmin(max_frames_file, number_of_frames_needed - k * max_frames_file);

				std::string temp_str = std::to_string((uint32_t)k);

				std::string relavtive_pth = RELATIVE_PATH;
				relavtive_pth.append("Simulation_info_");
				relavtive_pth.append(temp_str);
				relavtive_pth.append(".csv");

				std::ofstream file1(relavtive_pth, std::ios::out);

				if (!file1.is_open()) {
					std::cout << "Information file " << k << "failed!!\n";
					error_check = -1;
					// break;
				}
				else{
					file1 << Grid_Nodes << "," << PtsPerDim << ",";
						
					for (uint16_t d = 0; d < DIMENSIONS; d++){
						file1 << H_Mesh[0].dim[d] << "," << H_Mesh[Grid_Nodes - 1].dim[d] << ",";
					} 
						file1 << duration.count() << "\n";

					#if IMPULSE_TYPE == 0 || IMPULSE_TYPE ==1
						for (uint32_t i = k * max_frames_file + frames_init; i < k*max_frames_file + frames_in_file + frames_init - 1; i++) {
							file1 << time_vector[i].time << ",";
						}
						file1 << time_vector[k*max_frames_file + frames_in_file + frames_init - 1].time;

					#elif IMPULSE_TYPE == 2
						file1 << time_vector[k * max_frames_file + frames_init].time << ",";

						for (uint32_t i = k * max_frames_file + 1 + frames_init; i < k*max_frames_file + frames_init + frames_in_file; i++) {
							if(abs(time_vector[i].time - time_vector[i-1].time)>pow(10,-7)){
								file1 << time_vector[i].time << ",";
							}
							else if(i == k*max_frames_file + frames_in_file + frames_init - 1){
								if(time_vector[i].time != time_vector[i-1].time){
									file1 << time_vector[i].time;
								}
							}
						}
					#endif
					file1.close();
				}

				// Simulation Information
				relavtive_pth = RELATIVE_PATH;
				relavtive_pth.append("Mean_PDFs_");
				relavtive_pth.append(temp_str);
				relavtive_pth.append(".bin");

				std::ofstream myfile(relavtive_pth, std::ios::out | std::ios::binary);

				if(!myfile.is_open()){
					std::cout << "Simulation output file " << k << " failed!!" << std::endl;
					error_check = -1;
					// break;
				}
				else{
					myfile.write((char *)&store_PDFs[(k*max_frames_file + frames_init)*Grid_Nodes], sizeof(float) * frames_in_file * Grid_Nodes);
					myfile.close();
					std::cout << "Simulation output file " << k << " completed!" << std::endl;
				}
				
			}
		}

		saving_active = false;
	}

	delete[] H_Mesh;
	delete[] Param_dist;
	delete[] Parameter_Mesh;
	gpuError_Check(cudaDeviceReset());

	return error_check;
}