#ifndef __SIMULATION_PARAMETERS_CUH__
#define __SIMULATION_PARAMETERS_CUH__

#include "Constants.cuh"
#include "Sim_data.cuh"

int16_t BuildTimeVector(std::vector<Time_instants>& time_vector, double& deltaT, INT& ReinitSteps) {
// MODIFIABLE
// 1.- Time values for output + impulse vectors
	double t0, tF;

	// Read init. Time from terminal 
	bool get_answer = true;
	while (get_answer) {
		std::string terminalInput;

		std::cout << "Choose initial simulation time: ";
		std::cin >> terminalInput;

		if(!isNumeric(terminalInput)){std::cout << "Error: Non-numeric entries are not allowed. ";}
		else{
			t0 = std::stod(terminalInput);

			if (t0 == -1) {return -1;}
			
			if (t0 < 0) {
				std::cout << "You must choose a STRICTLY positive initial time.\n";
			}
			else {
				get_answer = false;
			}
		}
	}

	// Read final time from terminal
	get_answer = true;
	while (get_answer) {

		std::string terminalInput;
		std::cout << "Choose end simulation time: ";
		
		std::cin >> terminalInput;
		if(!isNumeric(terminalInput)){std::cout << "Error: Non-numeric entries are not allowed. ";}
		else{		
			tF = std::stod(terminalInput);

			if (tF == -1) {return -1;}
			
			if (tF <= t0) {
				std::cout << "Your final time must be STRICTLY larger than the initial time.\n";
			}
			else {
				get_answer = false;
			}
		}
	}

	// Get timestep
	get_answer = true; 
	while (get_answer) {

		std::string terminalInput;
		std::cout << "Choose time-step: ";
		std::cin >> terminalInput;

		if(!isNumeric(terminalInput)){std::cout << "Error: Non-numeric entries are not allowed. ";}
		else{
			deltaT = std::stod(terminalInput);
			if (deltaT == -1) {
				return -1;
			}
			
			if (deltaT == 0) {
				std::cout << "You must choose a STRICTLY positive timestep.\n";
			}
			else {
				get_answer = false;
			}
		}
	}

	// Get reinit steps
	get_answer = true;
	while (get_answer) {

		std::string terminalInput;
		std::cout << "Re-initialization steps?: ";
		std::cin >> terminalInput;

		if(!isNumeric(terminalInput)){std::cout << "Error: Non-numeric entries are not allowed. ";}
		else{

			ReinitSteps = std::stoi(terminalInput);

			if (ReinitSteps == -1) {
				return -1;
			}
			
			if (ReinitSteps == 0) {
				std::cout << "You must choose a STRICTLY positive number of steps.\n";
			}
			else {
				get_answer = false;
			}
		}
	}

	// Build the vector as such!
	while (t0 < tF + ReinitSteps * deltaT / 2) {						// THIS WAY, WE MAKE SURE THAT ROUND-OFF ERRORS ARE NOT ALLOWED!!
		time_vector.push_back({ t0, false});
		t0 = t0 + ReinitSteps * deltaT;
	}

// 2.- if there are impulses: SHOULD RECHEK AND SEE IF IT WORKS!
	#if (IMPULSE_TYPE != 0)
		#if (IMPULSE_TYPE == 1)
			for (short int j = 0; j < DiracDelta_impulseCount; j++){
				if(deltaImpulse_distribution_TIME[j] < tF){	time_vector.push_back({deltaImpulse_distribution_TIME[j], true});	}	// Include time instants where impulses take place
				else{std::cout << "WARNING: Some impulse time instants are larger than the simulation final time. Simulation will stop at tF.\n";break;}
			}
		#elif (IMPULSE_TYPE == 2)
			short int impulse_count = 0;

			while (impulse_count < SQUARE_jumpCountS){
				if(SQ_jumpCount_DIST_TIME[2 * impulse_count] < tF){	
					time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count], true});
					time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count], false});	
				}
				else{std::cout << "WARNING: Some forcing time instants are larger than the simulation final time. Simulation will stop at tF.\n";break;}

				if(SQ_jumpCount_DIST_TIME[2 * impulse_count + 1] < tF){
					time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count + 1], true});
					time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count + 1], false});
					impulse_count++;
				}
				else{std::cout << "WARNING: Some forcing time instants are larger than the simulation final time. Simulation will stop at tF.\n";break;}
			}		
		#else
			std::cout << "Error in 'Simulation_parameters.cuh'. You are choosing an unavailable jumpCount option. Go back to 'Classes.cuh' and re-check options for IMPULSE_TYPE.\n"
		#endif

		std::sort(time_vector.begin(), time_vector.end());					// re-sort the time values of the vector
		for (unsigned i = 0; i < time_vector.size() - 1; i++) {				// Update dt
			double aux_dt = time_vector[i + 1].time - time_vector[i].time;	// auxiliary dt for next computation

			if (aux_dt > (double)pow(10,-8)) {
				deltaT = fmin(deltaT, aux_dt / ReinitSteps);				// update dt if the impulse time does not match the "normal" resampling instants
			}
			else if (aux_dt < (double)-pow(10, -8)) {
				deltaT = fmin(deltaT, -aux_dt / ReinitSteps);
			}
			else if (time_vector[i].impulse == false && time_vector[i+1].impulse == true) {
				std::swap(time_vector[i], time_vector[i + 1]);		// swap the impulse times so that the one with the impulse is set before
			}
			else if (time_vector[i].impulse == time_vector[i+1].impulse) {
				time_vector.erase(time_vector.begin() + i);	
				i--;														// erase the repeated impulse time so that the one with the impulse is set before
			}
		}
		
	#endif
	return 0;
}

#endif