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

#ifndef __IVPSOLVER_HPP__
#define __IVPSOLVER_HPP__

#include "include/headers.hpp"
#include "include/utils/numeric_defs.hpp"
#include "include/indicators/progress_bar.hpp"

#include "adaptiveMeshRefinement/Adapt_Mesh.hpp"
#include "mesh/Domain.hpp"
#include "mesh/Particle.hpp"
#include "probabilityDistributions/Probability.hpp"
#include "interpolation/Interpolation.hpp"
#include "include/PointSearch.cuh"
#include "integrators/Impulse_transformations.hpp"
#include "integrators/Integrator.hpp"

#include "Simulation_parameters.hpp"
#include "Sim_data.hpp"

class ivpSolver{
private:
	//Domain where the PDF will evolve (positively invariant set)
	cartesianMesh __problem_domain;
	cartesianMesh __particle_bounding_box;
	
	// Distributions for the model parameters
	Distributions __initial_condition_distributions[PHASE_SPACE_DIMENSIONS];
	Distributions __parameter_distributions[PARAM_SPACE_DIMENSIONS];
	
	// Time vector, impulse information, timestep and effective timestep
	std::vector<Time_instants> __reinitialization_info;
	double __delta_t;
	intType __reinitialization_steps, __storage_steps;

	LogSimulation __simulation_log;

	// Final simulation storage
	std::vector<floatType> __simulation_storage;


public:
	// Set the default constructor. Parametric constructor won't be needed!
	ivpSolver();
	~ivpSolver();

	// This method is used to build the domain in which the simulation will take place
	int16_t buildDomain();

	// Use the time information 
	int16_t buildTimeVec();

	// Build the Distribution information for all random parameters
	int16_t buildDistributions();
	
	// This function contains the most important function of them all: The full numerical method!
	int16_t evolvePDF(const cudaDeviceProp& D_Properties);

	int16_t writeFramesToFile(const double& simulationDuration);
};

#endif