#include "Classes.cuh"
#include "Simulation_parameters.cuh"
#include "Dynamics.cuh"
#include "Parameter_distributions.cuh"
#include "Adapt_Mesh.cuh"
#include "Mat_Ops.cuh"
#include "Impulse_transformations.cuh"

#include "Debugging.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

__host__ int PDF_ITERATIONS(std::vector<double>* store_PDFs,
							const std::vector<Param_vec>* Parameter_Mesh,
							const gridPoint* H_Mesh,
							thrust::host_vector<double>* H_PDF,
							const int& LvlFine,
							const int& LvlCoarse,
							const int& PtsPerDim,
							const int& Grid_Nodes,
							const std::vector<Time_Impulse_vec>	time_vector,
							const float& deltaT,
							const int& ReinitSteps);

//--------------------------------------------------------------------------------------------- //


/// @brief This function computes the evolution of an initial PDF (defined in "Parameter_distributions.cuh")
///  according to the dynamics parameters (also defined in "Parameter_distributions.cuh"), with some 
/// given simulation parameters (given in "Simulation_parameters.cuh"). 
/// @return An integer (0 if everything was OK, -1 if the simulaion was forced to stop for some reason)
int PDF_EVOLUTION() {
	std::cout << "You must choose: \n - FINEST / COARSEST MESH LEVEL \n - If you wish to include IMPULSE terms \n - FINAL time and TIMESTEP";
	std::cout << "\n - Reinitialization Steps \n - SAMPLES PER PARAMETER \n";
	std::cout << "(REMINDER) You can type \"-1\" to  exit the program while selecting the simulation parameters.\n";
	// Pre-Simulation Data initialization -------------------------------------------------------------------------------

	// ----------------------------------------------------------------------------------------------- //
	// ---------------------------------- OBTAIN INFO FROM TERMINAL ---------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	int LvlFine, LvlCoarse = 0;
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

	const int PtsPerDim  = (int)powf(2, LvlFine);
	const int Grid_Nodes = (int)powf(PtsPerDim, DIMENSIONS);

	// Build H_Mesh here (TO BE MODIFIED FOR OTHER DIMENSIONS)
	const gridPoint Domain_Center = {0.5, 0.5};
	const gridPoint Domain_Radius = {12, 12};

	gridPoint* H_Mesh = new gridPoint[Grid_Nodes]; // maybe it could be implemented as constant memory?

	// GENERAL DIMENSION Cartesian coordinate grid
	for (unsigned int i = 0; i < Grid_Nodes; i++){
		for (unsigned int d = 0; d < DIMENSIONS; d++){
			unsigned int j = floor(positive_rem(i, pow(PtsPerDim, d + 1))/pow(PtsPerDim, d));
			H_Mesh[i].dim[d] = ((double) j / (PtsPerDim - 1) - Domain_Center.dim[d]) * Domain_Radius.dim[d]; 
		}
	}

	// Time simulation data Definition: -------------------------------------------------------------------------------
	int			ReinitSteps;
	double		deltaT;
	std::vector<Time_Impulse_vec> time_vector;
	
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	// ----------------------------------------------------------------------------------------------- //

	int error_check = Simul_Data_Def(time_vector, deltaT, ReinitSteps);
	if (error_check == -1){
		std::cout << "Exiting simulation.\n";
		return error_check;
	}

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
	// 1.- PARAMETER H_MESH biuld up
	int n_samples[PARAM_DIMENSIONS];						// number of samples per parameter
	
	// ----------------------------------------------------------------------------------------------- //
	// ---------------------------------- OBTAIN INFO FROM TERMINAL ---------------------------------- //
	// ----------------------------------------------------------------------------------------------- //
	for (int k = 0; k < PARAM_DIMENSIONS; k++) {
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
	int Random_Samples = 1;
	for (unsigned int i = 0; i < PARAM_DIMENSIONS; i++){
		Random_Samples *= n_samples[i];
	}

	std::vector<Param_vec>	Parameter_Mesh(Random_Samples);					// Full parameter array
	double 					Dist_Params[PARAM_DIMENSIONS * 2];
	char 					Dist_Names[PARAM_DIMENSIONS];

	// 1st RV mean and variance
	Dist_Names[0]  = 'N';
	Dist_Params[0] = 0.2;
	Dist_Params[1] = 0.02;

	// 2nd RV mean and variance
	Dist_Names[1]  = 'N';
	Dist_Params[2] = 3;
	Dist_Params[3] = 0.3;

	RANDOMIZE(n_samples, Random_Samples, &Parameter_Mesh, Dist_Params, Dist_Names);	

	std::cout << "Total number of random samples: " << Random_Samples << ".\n";

	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------
	// 2.- INITIAL PDF build up
	thrust::host_vector<double> H_PDF(Grid_Nodes);	 // PDF values at the fixed, high-res grid (CPU)

	double IC_dist_params[DIMENSIONS * 2];

	// one pair per dimension
	IC_dist_params[0] = 1.75;  // mean
	IC_dist_params[1] = sqrt(0.025); // var

	IC_dist_params[2] = 0.00;  // mean
	IC_dist_params[3] = sqrt(0.025); // var

	PDF_INITIAL_CONDITION(PtsPerDim, H_Mesh, H_PDF, IC_dist_params); // initialize the grid and the PDF at the grid nodes (change so as to change the parameters as well)

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// 3.- Evolution simulation

	std::vector<double>	store_PDFs(0);		 // H_PDF storage for post-processing

	auto start = std::chrono::high_resolution_clock::now();

	
	error_check = PDF_ITERATIONS(&store_PDFs, &Parameter_Mesh, H_Mesh, &H_PDF, LvlFine, LvlCoarse, PtsPerDim, Grid_Nodes, time_vector, deltaT, ReinitSteps);
	// To do: CREATE TREE STRUCT FOR THE AMR-POINTS + conservative scheme
	if (error_check == -1){
		std::cout << "An error has occured. Exiting simulation.\n";
		return error_check;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start; // duration

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// ---------------------- STORAGE INTO COMPUTER HDD for post-processing -----------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
	char ans;
	std::cout << "Simulation time: " << duration.count() << " seconds. Memory to be written: " << (float)store_PDFs.size() * sizeof(double) / 1000000 << " MB. Write? (Y=Yes, N=no)\n";
	std::cin >> ans;

	if ((ans == 'Y') || (ans == 'y')) {
		std::cout << "Writing to memory...please wait (it may take over 10 seconds)\n";

		// Simulation Information
		std::ofstream file1;
		file1.open("Simulation_Info.csv");
		if (file1.is_open()) {
			//file1 << "Total Grid Points," << "Points per dimension," << "Grid X min," << "Grid X max," << "Grid Y min," << "Grid Y max," << "Time values," << "Simulation cost" << "t0" << "deltaT" << "Reinitialization Steps" << "\n";
			file1 << Grid_Nodes << "," << PtsPerDim << "," << H_Mesh[0].dim[0] << "," << H_Mesh[Grid_Nodes - 1].dim[0]
				<< "," << H_Mesh[0].dim[1] << "," << H_Mesh[Grid_Nodes - 1].dim[1] << "," << time_vector.size() << "," << duration.count() << "," << "\n";

			for (int i = 0; i < time_vector.size() - 1; i++) {
				file1 << time_vector[i].time << ",";
			}
			file1 << time_vector[time_vector.size() - 1].time;
			file1.close();
		}
		else {
			std::cout << "Fail.\n";
		}

		// Simulation Data
		std::ofstream myfile;
		myfile.open("Mean_PDFs.csv");
		if (myfile.is_open()) {
			for (int k = 0; k < store_PDFs.size() - 1; k++) {
				myfile << store_PDFs[k] << ",";
			}
			myfile << store_PDFs[store_PDFs.size() - 1];
			myfile.close();
			std::cout << "Completed!\n";
		}
		else {
			std::cout << "Failed!!\n";
		}
	}

	delete[] H_Mesh;

	return 0;
}


//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//------------------------ MAIN COMPUTATION FUNCTIONS -----------------------------------------
//---------------------------------------------------------------------------------------------
// 1.- Runge-Kutta integrator (particle + value)   --------------------------------------------
// 2.- Iterations of the Liouville-Equation solver --------------------------------------------
//---------------------------------------------------------------------------------------------

/// @brief This function computes the advection of the particles created by AMR.
/// 		It uses the RK4 scheme for the spatial variables advection and the Simpson rule for the exponential integral.
/// @param H_Mesh Particle location (spatial variables)
/// @param PDF PDF value at the corresponding particle location
/// @param parameters Parameters to be used for the vector field and its corresponding divergence function
/// @param t0 Inital time for starting the simulation
/// @param deltaT Time step used in the simulation
/// @param ReinitSteps Number of steps before needing a re-interpolation
/// @param Adapt_Points Number of particles as computed by the AMR scheme
/// @param Random_Samples Number of random parameter samples
/// @return 
__global__ void RungeKutta(	gridPoint* 			H_Mesh,
							double* 			PDF,
							const Param_vec* 	parameters,
							double				t0,
							const double		deltaT,
							const int			ReinitSteps,
							const int			Adapt_Points,
							const int			Random_Samples) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Adapt_Points * Random_Samples) {
		int steps = 0;

		// So, the total amount of advections are going to be: (n� particles x n� of samples)
		const int i_sample = floorf(i / Adapt_Points);

		// AUXILIARY DATA TO RUN THE ITERATIONS
		gridPoint k0, k1, k2, k3, aux;
		double	  Int1, Int2, Int3;

		gridPoint x0 	= H_Mesh[i]; 	// register storing the initial particle dim
		double Int_PDF 	= PDF[i];	// register storing the initial particle value

		while (steps < ReinitSteps) {

			// Particle flow
			k0 = VECTOR_FIELD(x0, t0, parameters[i_sample]);

			aux = Mult_by_Scalar(deltaT / 2, k0);
			k1 = VECTOR_FIELD(x0 + aux, t0 + deltaT / 2, parameters[i_sample]);

			aux = Mult_by_Scalar(deltaT / 2, k1);
			k2 = VECTOR_FIELD(x0 + aux, t0 + deltaT / 2, parameters[i_sample]);

			aux = Mult_by_Scalar(deltaT, k2);
			k3 = VECTOR_FIELD(x0 + aux, t0 + deltaT, parameters[i_sample]);

			k1 = Mult_by_Scalar(2, k1);
			k2 = Mult_by_Scalar(2, k2);

			aux = x0 + Mult_by_Scalar(deltaT / 6, (k0 + k3 + k1 + k2)); // New particle dim

			// Integration of PDF
			Int1 = DIVERGENCE_FIELD(x0, t0, parameters[i_sample]);

			x0 	 = Mult_by_Scalar(0.5, (x0 + aux));
			Int2 = DIVERGENCE_FIELD(x0, (2 * t0 + deltaT) / 2, parameters[i_sample]);

			Int3 = DIVERGENCE_FIELD(aux, t0 + deltaT, parameters[i_sample]);

			Int_PDF *= exp(-deltaT / 6 * (Int1 + 4 * Int2 + Int3)); // New particle value (change for the Hermite interpolation in the midpoint)

			// Reinit step
			x0 = aux;
			t0 += deltaT;
			steps++;
		}

		H_Mesh[i] 	= aux;
		PDF[i]		= Int_PDF;
	}
}


/// <summary>
/// 
/// This function computes the Liouville Eq. iterations from t0 to tF. Each iteration consists of the following steps:
/// 1st) Compute the AMR of the initial PDF. 
/// 2nd) Create as many particle families as random samples there are.
/// 3rd) Advect all particles via RungeKutta function. 
/// 4th) Interpolate all families of particles onto the underlying high-res. fixed grid. 
/// 5th) Compute ensemble mean.
/// 6th) Store final PDF into the iteration vector for further post-processing and/or evolution visualization.
/// 
/// </summary>
/// <param name="store_PDFs"> - Iteration storage (all PDF evolution "slides")</param>
/// <param name="Parameter_Mesh"> - Parameter values H_Mesh + joint PDF value </param>
/// <param name="H_Mesh"> - Initial, fixed, high-res. grid where we obtain the adapted particles</param>
/// <param name="H_PDF"> - Initial PDF values at the H_Mesh</param> 
/// <param name="LvlFine"> - Finest level in the AMR grids</param>
/// <param name="LvlCoarse"> - Coarsest level in the AMR grids</param>
/// <param name="PtsPerDim"> - Points per dimension</param>
/// <param name="Grid_Nodes"> - Total number of grid nodes</param>
/// <param name="time_vector"> - vector containing all the info about the time and implulses in the simulation</param>
/// <param name="deltaT"> - timestep for the RK4 scheme</param>
/// <param name="ReinitSteps"> - deltaT steps until next re-interpolation in high res. grid</param>
/// <returns></returns>
__host__ int PDF_ITERATIONS(std::vector<double>* store_PDFs,
							const std::vector<Param_vec>* Parameter_Mesh,
							const gridPoint* H_Mesh,
							thrust::host_vector<double>* H_PDF,
							const int& LvlFine,
							const int& LvlCoarse,
							const int& PtsPerDim,
							const int& Grid_Nodes,
							const std::vector<Time_Impulse_vec> time_vector,
							const float& deltaT,
							const int& ReinitSteps) {

//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
//------------------- INITIALIZATION OF VECTORS/ARRAYS THAT WILL BE USED ---------------------//
//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------//
	std::vector<gridPoint>	AdaptGrid(0);			// Particle positions to be used for simulation (corresponding dim from AMR)
	std::vector<double>		AdaptPDF(0);			// PDF value at the particle positions (corresponding values from AMR)
	std::vector<gridPoint>	Full_AdaptGrid(0);		// Final adapted grid (adapted grid x number of samples)
	std::vector<double>		Full_AdaptPDF(0);		// Final adapted PDF (adapted grid x number of samples)

	thrust::device_vector<gridPoint>	GPU_Part_Position;		// Particle positions (for the GPU)
	thrust::device_vector<double>		GPU_AdaptPDF;			// PDF value at Particle positions (for the GPU)
	thrust::device_vector<Param_vec>	GPU_Parameter_Mesh;		// Parameter H_Mesh array (for the GPU)

	const int Random_Samples = Parameter_Mesh->size();			// number of random parameter samples

	thrust::device_vector<gridPoint>	GPU_Mesh(H_Mesh, H_Mesh + Grid_Nodes);	// Fixed Grid Nodes (for the GPU)
	thrust::device_vector<double>		GPU_PDF = *H_PDF;						// PDF values at fixed Grid Nodes (for the GPU)

	// auxiliary variable that will be used for ensemble mean computation
	double Sum_Rand_Params = 0;
	for (int i = 0; i < Random_Samples; i++) {
		Sum_Rand_Params += Parameter_Mesh->at(i).Joint_PDF;
	}


// ------------------ DEFINITION OF THE INTERPOLATION VARIABLES AND ARRAYS ------------------ //
	int Adapt_Points, Total_Particles, MaxNeighborNum;

	const double disc_X = (H_Mesh[1].dim[0] - H_Mesh[0].dim[0]);	// H_Mesh discretization size
	const double search_radius = 4.75 * disc_X;						// max radius to search ([4,6] appears to be optimal)

	const int	 max_steps = 1000;		 // max steps at the Conjugate Gradient (CG) algorithm
	const double in_tolerance = pow(10, -8); // CG stop tolerance

	thrust::device_vector<int>		GPU_Index_array;
	thrust::device_vector<double>	GPU_Mat_entries;
	thrust::device_vector<int>		GPU_Num_Neighbors;

// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// ---------------------------------- SIMULATION TIME!! ---------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

// -------------------- Store the 1st PDF (Init. Cond.) -----------------------------------
	store_PDFs->insert(store_PDFs->end(), H_PDF->begin(), H_PDF->end());

	// ------------------------------------------------------------------------------------
	int j = 0;

	while (j < time_vector.size() - 1) {

		auto start_2 = std::chrono::high_resolution_clock::now();

		float	t0 = time_vector[j].time,
			tF = time_vector[j + 1].time;

		std::cout << "/-------------------------------------------------------------------/\n";
		if (t0 == tF) {
			std::cout << "RVT transformation at time: " << t0 << "\n";
		}
		else {
			std::cout << "Simulation time: " << t0 << " to " << tF << "\n";
		}

// 1.- Initial step Adaptive H_Mesh Refinement. First store the initial PDF with AMR performed
		ADAPT_MESH_REFINEMENT_nD(*H_PDF, &AdaptPDF, H_Mesh, &AdaptGrid, LvlFine, LvlCoarse, PtsPerDim);

	// 1.1.- COMPUTE THE TRANSFORMATION OF THE PDF (IF THERE IS ONE)
		if (time_vector[j].impulse && (time_vector[j].samples[0] > 0 || time_vector[j].samples[1] > 0)) {
			// compute transformaton
			//std::cout << "Computing impulse transformation.\n";

			int success_impulse = IMPULSE_TRANSFORM_PDF(H_Mesh, 
														&AdaptGrid, 
														H_PDF, 
														&AdaptPDF, 
														time_vector[j], 
														Grid_Nodes, 
														PtsPerDim);

			if (success_impulse != 0) {
				//std::cout << "Something went wrong...\n";
				std::cin.get();
				return success_impulse;
			}
			else {
				//std::cout << "Transformation done...continuing with Liouville PDE\n";
				AdaptGrid.clear();
				AdaptPDF.clear();
			}
		}
	// 1.2.- Error?
		else if (time_vector[j].impulse) {
			std::cout << "Check you impulse samples...you need at least one sample for the impulse.\n";
			break;
		}
	// 1.3.- COMPUTE THE SMOOTH EVOLUTION VIA LIOUVILLE GIBBS / CONTINUITY EQUATION
		else {

			Adapt_Points = AdaptGrid.size();
			Total_Particles = Adapt_Points * Random_Samples;

			// 1.2.- Append the optimal particles once per sample!
			for (int k = 0; k < Random_Samples; k++) {
				Full_AdaptGrid.insert(Full_AdaptGrid.end(), AdaptGrid.begin(), AdaptGrid.end());
				Full_AdaptPDF.insert(Full_AdaptPDF.end(), AdaptPDF.begin(), AdaptPDF.end());
			}

			// Upload information to the GPU
			GPU_AdaptPDF 		= Full_AdaptPDF;
			GPU_Part_Position 	= Full_AdaptGrid;
			GPU_Parameter_Mesh 	= *Parameter_Mesh;

			// Clear CPU info about the AMR procedure: no longer needed
			AdaptGrid.clear();
			AdaptPDF.clear();
			Full_AdaptGrid.clear();
			Full_AdaptPDF.clear();

			std::cout << "Size of relevant PDF points (per sample): " << Adapt_Points << "\n";	// this allows to check if the info is passed to the GPU correctly

			MaxNeighborNum = (int)fminf(200, Adapt_Points);				// maximum neighbors to search


// ------------------ RESIZING OF THE INTERPOLATION MATRIX ------------------ //
			GPU_Index_array.resize(MaxNeighborNum * Total_Particles);
			GPU_Mat_entries.resize(MaxNeighborNum * Total_Particles);
			GPU_Num_Neighbors.resize(Total_Particles);
// -------------------------------------------------------------------------- //
			// Determine threads and blocks for the simulation
			int Threads = (int)fminf(THREADS_P_BLK, Total_Particles);
			int Blocks  = (int)floorf((Total_Particles - 1) / Threads) + 1;

// ------------------------------------------------------------------------------------ //
// -------------------------- POINT ADVECTION ----------------------------------------- //
// ------------------------------------------------------------------------------------ //
			// Using RK4 for time integration of characteristic curves
			RungeKutta << <Blocks, Threads >> >(raw_pointer_cast(&GPU_Part_Position[0]), 
												raw_pointer_cast(&GPU_AdaptPDF[0]),
												raw_pointer_cast(&GPU_Parameter_Mesh[0]), 
												t0, 
												deltaT, 
												ReinitSteps, 
												Adapt_Points, 
												Random_Samples);
			gpuError_Check(cudaDeviceSynchronize()); // Here, the entire H_Mesh points (those that were selected) and PDF points (same) have been updated.

// ----------------------------------------------------------------------------------- //
// -------------------------- INTERPOLATION ------------------------------------------ //
// ----------------------------------------------------------------------------------- //
	// 1.- Build Matix in GPU (indexes, dists and neighbors) Using Exahustive search...
			Exh_PP_Search << <Blocks, Threads >> > (raw_pointer_cast(&GPU_Part_Position[0]), 
													raw_pointer_cast(&GPU_Part_Position[0]), 
													raw_pointer_cast(&GPU_Index_array[0]),
													raw_pointer_cast(&GPU_Mat_entries[0]), 
													raw_pointer_cast(&GPU_Num_Neighbors[0]), 
													MaxNeighborNum, 
													Adapt_Points, 
													Total_Particles, 
													search_radius);
			gpuError_Check(cudaDeviceSynchronize());

	// 2.- Iterative solution (Conjugate Gradient) to obtain coefficients of the RBFs
			thrust::device_vector<double>	GPU_lambdas(Total_Particles);	// solution vector (RBF weights)
			thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0);		// this will serve as the initial condition

			int err = CONJUGATE_GRADIENT_SOLVE<double>( GPU_lambdas, 
														GPU_Index_array, 
														GPU_Mat_entries, 
														GPU_Num_Neighbors, 
														GPU_AdaptPDF, 
														Total_Particles, 
														MaxNeighborNum, 
														max_steps, 
														in_tolerance);
			if (err == -1) { return err; }

	// 3.- Multiplication of matrix-lambdas to obtain updated grid nodes
			thrust::fill(GPU_PDF.begin(), GPU_PDF.end(), 0);	// PDF is reset to 0, so that we may use atomic adding

			// I'M GOING TO FIND THE NEAREST GRID NODES TO EACH PARTICLE
			for(unsigned int s = 0; s < Random_Samples; s++){
				// Re-define Threads and Blocks
				Threads = fminf(THREADS_P_BLK, Adapt_Points);
				Blocks  = floorf((Adapt_Points - 1) / Threads) + 1;

				auto start_2 = std::chrono::high_resolution_clock::now();
				
				RESTART_GRID_FIND_GN<<< Blocks, Threads >>>(raw_pointer_cast(&GPU_Part_Position[0]),
															raw_pointer_cast(&GPU_PDF[0]),
															raw_pointer_cast(&GPU_lambdas[0]),
															raw_pointer_cast(&GPU_Mesh[0]),
															raw_pointer_cast(&GPU_Parameter_Mesh[0]),
															search_radius,
															H_Mesh[0],
															disc_X,
															PtsPerDim,
															Adapt_Points,
															s);
				gpuError_Check(cudaDeviceSynchronize());

				auto end_2 = std::chrono::high_resolution_clock::now();

				std::chrono::duration<float> duration_2 = end_2 - start_2;
			}

			// Correction of any possible negative PDF values
				// Re-define Threads and Blocks
				Threads = fminf(THREADS_P_BLK, Grid_Nodes);
				Blocks  = floorf((Grid_Nodes - 1) / Threads) + 1;
			CORRECTION<<<Blocks, Threads>>>(raw_pointer_cast(&GPU_PDF[0]), Grid_Nodes);
			gpuError_Check(cudaDeviceSynchronize());

			thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1 / Sum_Rand_Params * _1); // we use the thrust::placeholders here (@ the last input argument)

			*H_PDF = GPU_PDF; // Send back to CPU
		}

		// Store info in cumulative variable
		store_PDFs->insert(store_PDFs->end(), H_PDF->begin(), H_PDF->end());
		j++;

		auto end_2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<float> duration_2 = end_2 - start_2;
		std::cout << "Liouville iteration took " << duration_2.count() << " seconds\n";
		std::cout << "/-------------------------------------------------------------------/\n";
	}
	return 0;
}