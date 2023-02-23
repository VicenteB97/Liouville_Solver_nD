#ifndef __SIMULATION_PARAMETERS_CUH__
#define __SIMULATION_PARAMETERS_CUH__

#include "Classes.cuh"

/// <summary>
/// 
/// </summary>
/// <param name="time_Vec"></param>
/// <param name="deltaT"></param>
/// <param name="ReinitSteps"></param>
int Simul_Data_Def(std::vector<Time_Impulse_vec>& time_Vec, double& deltaT, int& ReinitSteps) {
// MODIFIABLE
// 1.- Time values for output + impulse vectors
	double t0 = 0;
	double tF;

	std::cout << "Choose end simulation time: ";
	std::cin >> tF;
	if (tF == -1){
		return -1;
	}

	std::cout << "Choose time-step: ";
	std::cin >> deltaT;
	if (deltaT == -1){
		return -1;
	}
	std::cout << "Re-initialization steps? (Ideally 2 or 4): ";
	std::cin >> ReinitSteps;
	if (ReinitSteps == -1){
		return -1;
	}

	while (t0 < tF + ReinitSteps * deltaT) {

		time_Vec.push_back({ t0, false, 0 , 0 , 0 });
		t0 = t0 + ReinitSteps * deltaT;
	}

// 2.- if there are impulses:

	if (IMPULSE) {
		
		#if (CASE == 1)//	  time | Imp| mean_vec| st. dev.| samples
			time_Vec.push_back({ 15, true,	{-0.5} ,	0.01 , 30 });	// Include time instants where impulses take place
			time_Vec.push_back({ 25, true,	{-0.5} ,	0.01 , 30 });	// Include time instants where impulses take place
			time_Vec.push_back({ 35, true,	{-0.5} ,	0.01 , 30 });	// Include time instants where impulses take place
		#endif
		#if(CASE == 2)	//		time | Imp | mean_vec  |   st. dev. | 	samples
			time_Vec.push_back({ 0.6, true,	{0,2.5} ,	{0, sqrt(0.02)} , {1,20} });	// Include time instants where impulses take place
			time_Vec.push_back({ 1.2, true,	{0,1} ,		{0, sqrt(0.02)} , {1,20} });	// Include time instants where impulses take place
			time_Vec.push_back({ 2.4, true,	{0,1.2} ,	{0, sqrt(0.02)} , {1,20} });	// Include time instants where impulses take place
		#endif

		std::sort(time_Vec.begin(), time_Vec.end());							// re-sort the time values of the vector
		for (unsigned int i = 0; i < time_Vec.size() - 1; i++) {				// Update dt
			double aux_dt = time_Vec[i + 1].time - time_Vec[i].time;			// auxiliary dt for next computation

			if (aux_dt > pow(10,-8)) {
				deltaT = fmin(deltaT, aux_dt / ReinitSteps);					// update dt if the impulse time does not match the "normal" resampling instants
			}
			else if (aux_dt < -pow(10, -8)) {
				deltaT = fmin(deltaT, -aux_dt / ReinitSteps);
			}
			else if (time_Vec[i].impulse == false && time_Vec[i+1].impulse == true) {
				std::swap(time_Vec[i],time_Vec[i + 1]);							// swap the impulse times so that the one with the impulse is set before
			}
		}
	}
	return 0;
}

#endif