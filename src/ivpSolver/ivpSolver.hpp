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
#include "interpolation/Remeshing.hpp"
#include "include/PointSearch.hpp"
#include "integrators/Impulse_transformations.hpp"
#include "integrators/Integrator.hpp"
#include "terminal/terminal.hpp"

#include "Simulation_parameters.hpp"
#include "Sim_data.hpp"

class ivpSolver{
private:
	//Domain where the PDF will evolve (positively invariant set)
	cartesianMesh m_problemDomain;
	cartesianMesh m_particleBoundingBox;
	
	// Distributions for the model parameters
	Distributions __initial_condition_distributions[PHASE_SPACE_DIMENSIONS];
	Distributions m_parameterDistributions[PARAM_SPACE_DIMENSIONS];
	
	// Time vector, impulse information, timestep and effective timestep
	std::vector<Time_instants> m_reinitializationInfo;
	double __delta_t;
	intType __reinitialization_steps, m_storageSteps;

	LogSimulation m_simulationLog;

	// Final simulation storage
	std::vector<floatType> m_simulationStorage;

	// terminal for current launch
	terminal m_terminal;


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
	int16_t evolvePDF();

	int16_t writeFramesToFile(const double& simulationDuration);
};

#endif