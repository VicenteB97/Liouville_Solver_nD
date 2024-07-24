#ifndef __SIMULATION_PARAMETERS_CUH__
#define __SIMULATION_PARAMETERS_CUH__

#include "headers.cuh"
#include "checkers/input_type_check.cuh"
#include "utils/error_msgs.cuh"
#include "utils/numeric_defs.cuh"
#include "Case_definition.cuh"
#include "Sim_data.cuh"

// Time + impulse: ----------------------------------------------
class Time_instants {
public:
    double 	time;
    bool 	impulse;

	bool operator < (const Time_instants& other) const;
};

int16_t BuildTimeVector(std::vector<Time_instants>& time_vector, double& deltaT, intType& ReinitSteps);

#endif