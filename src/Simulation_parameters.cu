#include <./include/Simulation_parameters.cuh>

bool Time_instants::operator < (const Time_instants& other) const {
	return (time < other.time);
}

int16_t BuildTimeVector(std::vector<Time_instants>& time_vector, double& deltaT, intType& ReinitSteps) {
	// MODIFIABLE
	// 1.- Time values for output + impulse vectors
	double t0, tF;

	// Read init. Time from terminal 
	bool get_answer = true;
	std::string terminalInput;

	while (get_answer) {

		std::cout << "Choose initial simulation time: ";
		std::cin >> terminalInput;

		errorCheck(doubleCheck(get_answer, terminalInput, INIT_TIME_ERR_MSG, -1E-6, 0))
	}
	t0 = std::stod(terminalInput);

	// Read final time from terminal
	get_answer = true;
	while (get_answer) {
		std::cout << "Choose end simulation time: ";
		std::cin >> terminalInput;

		errorCheck(doubleCheck(get_answer, terminalInput, END_TIME_ERR_MSG, 0, t0))
	}
	tF = std::stod(terminalInput);

	// Get timestep
	get_answer = true;
	while (get_answer) {

		std::cout << "Choose time-step: ";
		std::cin >> terminalInput;

		errorCheck(doubleCheck(get_answer, terminalInput, TIMESTEP_ERR_MSG, 0, 0))
	}
	deltaT = std::stod(terminalInput);

	// Get reinit steps
	get_answer = true;
	while (get_answer) {

		std::cout << "Re-initialization steps?: ";
		std::cin >> terminalInput;

		errorCheck(intCheck(get_answer, terminalInput, REINIT_ERR_MSG, 0, 1))
	}
	ReinitSteps = std::stoi(terminalInput);

	time_vector.resize((double) (tF-t0) / (ReinitSteps * deltaT) + 1);
	for(uintType i = 0; i < time_vector.size(); i++){
		time_vector[i] = {(double) t0 + i*ReinitSteps * deltaT, false};
	}


	// 2.- if there are impulses:
#if (IMPULSE_TYPE != 0)
	#if (IMPULSE_TYPE == 1)
	for (short int j = 0; j < DiracDelta_impulseCount; j++) {
		if (deltaImpulse_distribution_TIME[j] < tF) { time_vector.push_back({ deltaImpulse_distribution_TIME[j], true }); }	// Include time instants where impulses take place
		else { std::cout << "WARNING: Some impulse time instants are larger than the simulation final time. Simulation will stop at tF.\n"; break; }
	}
	#elif (IMPULSE_TYPE == 2)
	short int impulse_count = 0;

	while (impulse_count < SQUARE_jumpCountS) {
		if (SQ_jumpCount_DIST_TIME[2 * impulse_count] < tF) {
			time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count], true });
			time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count], false });
		}
		else { std::cout << "WARNING: Some forcing time instants are larger than the simulation final time. Simulation will stop at tF.\n"; break; }

		if (SQ_jumpCount_DIST_TIME[2 * impulse_count + 1] < tF) {
			time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count + 1], true });
			time_vector.push_back({ SQ_jumpCount_DIST_TIME[2 * impulse_count + 1], false });
			impulse_count++;
		}
		else { std::cout << "WARNING: Some forcing time instants are larger than the simulation final time. Simulation will stop at tF.\n"; break; }
	}
	#else
	std::cout << "Error in 'Simulation_parameters.cuh'. You are choosing an unavailable jumpCount option. Go back to 'Classes.cuh' and re-check options for IMPULSE_TYPE.\n"
	#endif

		std::sort(time_vector.begin(), time_vector.end());					// sort the time values of the vector
	for (unsigned i = 0; i < time_vector.size() - 1; i++) {				// Update dt
		double aux_dt = time_vector[i + 1].time - time_vector[i].time;	// auxiliary dt for next computation

		if (aux_dt > (double)pow(10, -8)) {
			deltaT = fmin(deltaT, aux_dt / ReinitSteps);				// update dt if the impulse time does not match the "normal" resampling instants
		}
		else if (aux_dt < (double)-pow(10, -8)) {
			deltaT = fmin(deltaT, -aux_dt / ReinitSteps);
		}
		else if (time_vector[i].impulse == false && time_vector[i + 1].impulse == true) {
			std::swap(time_vector[i], time_vector[i + 1]);				// swap the impulse times so that the one with the impulse is set before
		}
		else if (time_vector[i].impulse == time_vector[i + 1].impulse) {
			time_vector.erase(time_vector.begin() + i);
			i--;														// erase the repeated impulse time so that the one with the impulse is set before
		}
	}

#endif
	return 0;
}
