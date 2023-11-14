#ifndef __SIMULATION_PARAMETERS_CUH__
#define __SIMULATION_PARAMETERS_CUH__

#include "Constants.cuh"
#include "Sim_data.cuh"

int16_t Simul_Data_Def(std::vector<Time_instants>& time_vector, double& deltaT, INT& ReinitSteps) {
// MODIFIABLE
// 1.- Time values for output + impulse vectors
	double t0 = Time_0, tF;

	bool get_answer = true;
	while (get_answer) {
		std::cout << "Choose end simulation time: ";
		std::cin >> tF;
		if (tF == -1) {
			return -1;
		}
		else if (tF <= t0) {
			std::cout << "You must choose a STRICTLY positive time value.\n";
		}
		else {
			get_answer = false;
		}
	}

	get_answer = true; 
	while (get_answer) {
		std::cout << "Choose time-step: ";
		std::cin >> deltaT;
		if (deltaT == -1) {
			return -1;
		}
		else if (deltaT == 0) {
			std::cout << "You must choose a STRICTLY positive timestep.\n";
		}
		else {
			get_answer = false;
		}
	}

	get_answer = true;
	while (get_answer) {
		std::cout << "Re-initialization steps? (Ideally 2 or 4): ";
		std::cin >> ReinitSteps;
		if (ReinitSteps == -1) {
			return -1;
		}
		else if (ReinitSteps == 0) {
			std::cout << "You must choose a STRICTLY positive number of steps.\n";
		}
		else {
			get_answer = false;
		}
	}

	while (t0 < tF + ReinitSteps * deltaT / 2) {						// THIS WAY, WE MAKE SURE THAT ROUND-OFF ERRORS ARE NOT ALLOWED!!
		time_vector.push_back({ t0, false});
		t0 = t0 + ReinitSteps * deltaT;
	}

// 2.- if there are impulses:

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