#ifndef __IMPULSE_TRANSFORMATIONS_CUH__
#define __IMPULSE_TRANSFORMATIONS_CUH__


#include "Classes.cuh"
#include "Mat_Ops.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

/// <summary>
/// 
/// </summary>
/// <param name="Particle_Positions"></param>
/// <param name="impulse_strengths"></param>
/// <param name="Num_Particles_per_sample"></param>
/// <param name="Total_Particles"></param>
/// <returns></returns>
__global__ void TRANSFORM_PARTICLES(gridPoint*		Particle_Positions,
									const double*	impulse_strengths, 
									const int		Num_Particles_per_sample,
									const int		Total_Particles) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Total_Particles){

		const int j = floorf(i / Num_Particles_per_sample);				// current sample

		Particle_Positions[i].position[1] += impulse_strengths[j];		// we only have to add to the second variable!
	
	}
}

/// <summary>
/// This function computes the new PDF after a certain impulse transform according to the Random Variable Transformation Theorem (RVT). To be linked with the Liouville PDE solver.
/// </summary>
/// <param name="MESH"></param>
/// <param name="Adapt_MESH"></param>
/// <param name="PDF"></param>
/// <param name="Adapt_PDF"></param>
int IMPULSE_TRANSFORM_PDF(	const gridPoint*				MESH,			// Fixed Mesh
							const std::vector<gridPoint>*	Adapt_MESH,		// AMR-selected points
							thrust::host_vector<double>*	PDF,			// PDF in Mesh
							const std::vector<double>*		Adapt_PDF,		// PDF in AMR-selected points
							const Time_Impulse_vec			time,			// time-impulse information 
							const int						Grid_Nodes){	// Number of grid points 

// 0.- Create the impulse samples
		const int			num_samples = time.samples[1];		// only in the second variable (general will be done in the future)
		std::vector<double> impulse_weights(num_samples);		// the array where we store the Samples of the impulse random variable
		std::vector<double> impulse_values(num_samples);		// the array where we store the PDF of the impulse random variable
		double				sum_totalPDF = 0;					// The "total mass" of the PDF

		double mean = time.impulse_vec.position[1];

		int Adapt_Points = Adapt_PDF->size();

		if (num_samples > 1) {
			double var = time.impulse_std[1];

			double X0 = mean - 5 * sqrt(var);
			double XF = mean + 5 * sqrt(var);
			double dx = (XF - X0) / ((double)num_samples - 1);

			for (int i = 0; i < num_samples; i++) {
				impulse_values[i]	= X0 + i * dx;
				impulse_weights[i]	= 1 / sqrt(2 * M_PI * var) * exp(-0.5 / var * pow(impulse_values[i] - mean, 2));		// We assume it follows a normal distribution
				sum_totalPDF		+= impulse_weights[i];																	
			}
		}
		else {
			impulse_values[0]	= mean;
			impulse_weights[0]	= 1;
			sum_totalPDF		= 1;
		}

		std::vector<gridPoint>	Full_Adapt_Grid;
		std::vector<double>		Full_Adapt_PDF;

		for (int i = 0; i < num_samples; i++) {
			Full_Adapt_Grid.insert(Full_Adapt_Grid.end(), Adapt_MESH->begin(), Adapt_MESH->end());
			Full_Adapt_PDF.insert(Full_Adapt_PDF.end(), Adapt_PDF->begin(), Adapt_PDF->end());
		}

		const int Total_Points = Full_Adapt_PDF.size();

// 1.- Do the transformation of the points according to the info given by the time-impulse vector
		thrust::device_vector<gridPoint>	Particle_Positions	= Full_Adapt_Grid;		// To compute the transformations at the GPU
		thrust::device_vector<double>		PDF_Particles		= Full_Adapt_PDF;		// PDF of the transformations at the GPU
		thrust::device_vector<double>		Impulse_Strength	= impulse_values;		// To compute the transformation strengths at the GPU
		thrust::device_vector<double>		Impulse_Weights		= impulse_weights;		// Impulse weights at the GPU

		int Threads = fminf(1024, Total_Points);
		int Blocks  = floorf(Total_Points / Threads) + 1;

	// FOR SOME REASON, I'M ONLY WRITING IN SOME VALUES...NOT ALL OF THEM

	TRANSFORM_PARTICLES << <Blocks, Threads >> > (raw_pointer_cast(&Particle_Positions[0]), raw_pointer_cast(&Impulse_Strength[0]), Adapt_Points, Total_Points);
	gpuError_Check( cudaDeviceSynchronize() );


// 2.- RBF interpolation into the fixed grid
	
	// 2.1. - Find near particles
	const double	disc_X						= (MESH[1].position[0] - MESH[0].position[0]);
	const double	search_radius				= 7 * disc_X;									// max radius to search
	const int		MaxNeighborNum				= fminf(150, Adapt_Points);
	double			Iteration_information[2]	= { 0,0 };

	thrust::device_vector<int>		GPU_Index_array;
	thrust::device_vector<double>	GPU_Mat_entries;
	thrust::device_vector<int>		GPU_Num_Neighbors;

	GPU_Index_array.resize(MaxNeighborNum * Total_Points);
	GPU_Mat_entries.resize(MaxNeighborNum * Total_Points);
	GPU_Num_Neighbors.resize(Total_Points);

	Exh_PP_Search << <Blocks, Threads >> > (raw_pointer_cast(&Particle_Positions[0]), raw_pointer_cast(&Particle_Positions[0]), raw_pointer_cast(&GPU_Index_array[0]), raw_pointer_cast(&GPU_Mat_entries[0]), raw_pointer_cast(&GPU_Num_Neighbors[0]), MaxNeighborNum, Adapt_Points, Total_Points, search_radius);
	gpuError_Check( cudaDeviceSynchronize() );
	//std::cout << "Transformation Point Search: done\n";

	// 2.2.- Solve Conjugate Gradient system
	const int max_steps = 5000;
	const double in_tolerance = pow(10, -8);



	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------ //
	thrust::device_vector<double> GPU_lambdas;
	thrust::device_vector<double> GPU_R;
	thrust::device_vector<double> GPU_AUX;
	thrust::device_vector<double> GPU_AP;
	thrust::device_vector<double> GPU_P;

	// Lambdas: coefficients of the RBFs
	GPU_lambdas.resize(Total_Points);
	thrust::fill(GPU_lambdas.begin(), GPU_lambdas.end(), 0);	// this will serve as the initial condition

	// Auxiliary vectors for the Conjugate Gradient
	GPU_R.resize(Total_Points);
	GPU_AUX.resize(Total_Points);
	GPU_AP.resize(Total_Points);
	GPU_P.resize(Total_Points);


	int k = 1;

	// Auxiliary vectors and values
	double sq_error;
	bool flag = true;

	// Initialize Conjugate gradient method
	// Compute A * X0
	MATRIX_VECTOR_MULTIPLICATION << < Blocks, Threads >> > (raw_pointer_cast(&GPU_AUX[0]), raw_pointer_cast(&GPU_lambdas[0]), raw_pointer_cast(&GPU_Index_array[0]), raw_pointer_cast(&GPU_Mat_entries[0]), Total_Points, raw_pointer_cast(&GPU_Num_Neighbors[0]), MaxNeighborNum);
	gpuError_Check( cudaDeviceSynchronize() );

	// Compute R=B-A*X0
	DIFF_VECS << <Blocks, Threads >> > (raw_pointer_cast(&GPU_R[0]), raw_pointer_cast(&PDF_Particles[0]), raw_pointer_cast(&GPU_AUX[0]), Total_Points);
	gpuError_Check( cudaDeviceSynchronize() );

	double Alpha, R0_norm, aux, beta;

	GPU_P = GPU_R;

	while (flag) {
		// Alpha computation (EVERYTHING IS CORRECT!)
		// 1.1.- Compute AP=A*P
		MATRIX_VECTOR_MULTIPLICATION << < Blocks, Threads >> > (raw_pointer_cast(&GPU_AP[0]), raw_pointer_cast(&GPU_P[0]), raw_pointer_cast(&GPU_Index_array[0]), raw_pointer_cast(&GPU_Mat_entries[0]), Total_Points, raw_pointer_cast(&GPU_Num_Neighbors[0]), MaxNeighborNum);
		gpuError_Check( cudaDeviceSynchronize() );

		// 1.2.- Compute P'*AP
		thrust::transform(GPU_P.begin(), GPU_P.end(), GPU_AP.begin(), GPU_AUX.begin(), thrust::multiplies<double>());
		aux = thrust::reduce(thrust::device, GPU_AUX.begin(), GPU_AUX.end());

		// 1.3.- R'*R
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_AUX.begin(), thrust::multiplies<double>());
		R0_norm = thrust::reduce(thrust::device, GPU_AUX.begin(), GPU_AUX.end());

		Alpha = R0_norm / aux;

		// New X and R: (new, old, scalar, driving vec, total length)
			// 1.- Update Lambdas
		UPDATE_VEC << <Blocks, Threads >> > (raw_pointer_cast(&GPU_lambdas[0]), raw_pointer_cast(&GPU_lambdas[0]), Alpha, raw_pointer_cast(&GPU_P[0]), Total_Points);
		gpuError_Check( cudaDeviceSynchronize() );

		// 2.- Update residuals 
		UPDATE_VEC << <Blocks, Threads >> > (raw_pointer_cast(&GPU_R[0]), raw_pointer_cast(&GPU_R[0]), -Alpha, raw_pointer_cast(&GPU_AP[0]), Total_Points);
		gpuError_Check( cudaDeviceSynchronize() );

		// Compute residual norm
		thrust::transform(GPU_R.begin(), GPU_R.end(), GPU_R.begin(), GPU_AUX.begin(), thrust::multiplies<double>());
		sq_error = thrust::reduce(thrust::device, GPU_AUX.begin(), GPU_AUX.end()); // sum of its elements
		sq_error = sqrt(sq_error);

		if (sq_error < in_tolerance) {
			Iteration_information[0] = (int)k;
			Iteration_information[1] = sq_error;
			//std::cout << "Transformation convergence success. Iterations:" << Iteration_information[0] << " Error:" << Iteration_information[1] << "\n";
			flag = false;
		}
		else if (k > max_steps) {
			std::cout << "No transformation convergence was obtained. Last residual norm was: " << sq_error << "\n";
			std::cout << "/-------------------------------------------------------------------/\n";

			std::cin.get();
			return 0;
		}
		else {
			beta = sq_error * sq_error / R0_norm;

			UPDATE_VEC << <Blocks, Threads >> > (raw_pointer_cast(&GPU_P[0]), raw_pointer_cast(&GPU_R[0]), beta, raw_pointer_cast(&GPU_P[0]), Total_Points);
			gpuError_Check( cudaDeviceSynchronize() );
			k++;
		}
	}

	// 
	// 
	// 3.- Reinitialization

	// Multiplication of matrix-lambdas to obtain new points
	Threads = fminf(1024, Grid_Nodes);
	Blocks = floorf(Grid_Nodes / Threads) + 1;

	thrust::device_vector<double> GPU_PDF (Grid_Nodes);							// To compute the interpolation results at the GPU

	gridPoint* GPU_Mesh;
	cudaMalloc((void**)&GPU_Mesh, sizeof(gridPoint) * Grid_Nodes); // To compute the interpolation results at the GPU
	cudaMemcpy(GPU_Mesh, MESH, sizeof(gridPoint) * Grid_Nodes, cudaMemcpyHostToDevice);

	RESTART_GRID_II << <Blocks, Threads >> > (	raw_pointer_cast(&GPU_PDF[0]),
												GPU_Mesh, 
												raw_pointer_cast(&Particle_Positions[0]),
												raw_pointer_cast(&GPU_lambdas[0]), 
												raw_pointer_cast(&Impulse_Weights[0]),
												search_radius,
												Grid_Nodes,
												Adapt_Points, 
												Total_Points);
	gpuError_Check( cudaDeviceSynchronize() );

	//std::cout << "Transformation reinitialization: done\n";
	std::cout << "/-------------------------------------------------------------------/\n";

	thrust::transform(GPU_PDF.begin(), GPU_PDF.end(), GPU_PDF.begin(), 1 / sum_totalPDF * _1); // we use the thrust::placeholders here (@ the last input argument)

	thrust::copy(GPU_PDF.begin(), GPU_PDF.end(), PDF->begin());

	cudaFree(GPU_Mesh);

	return 1; // if this works
}

#endif