#ifndef __SIMULATION_PARAMETERS_HPP__
#define __SIMULATION_PARAMETERS_HPP__

#include "include/headers.hpp"
#include "include/checkers/input_type_check.hpp"
#include "Sim_data.hpp"

// Time + impulse: ----------------------------------------------
class Time_instants {
public:
    double 	time;
    bool 	impulse;

	bool operator < (const Time_instants& other) const;
};

int16_t BuildTimeVector(std::vector<Time_instants>& time_vector, double& deltaT, intType& ReinitSteps);

#endif