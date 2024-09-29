//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//														//
//		Written by: Vicente Jos� Bevia Escrig			//
//		Mathematics Ph.D. student (2020-2024) at:		//
//		Instituto de Matem�tica Multidisciplinar,		//
//		Universitat Polit�cnica de Val�ncia, Spain		//
//														//
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

#include "Interpolation.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
deviceFunction
void COOMatVecMultiplication_dvc::operator()(const uint64_t global_id) const {
	if (global_id >= totalLength) { return; }	// total length = adapt_points * total_sample_count

	// 1.- Compute A*X0										
	// 1.1.- Determine where my particles are!!
	const uintType i0 = global_id * maxNeighbors;	// where does my search index start

	floatType a = 0;	// auxiliary value for sum (the diagonal is always 1 in our case)
	uintType j = i0;
	while (matrixIdxs_dvc[j] != -1 && j < totalLength * maxNeighbors) {
		intType p = matrixIdxs_dvc[j];

		a += matrixEntries_dvc[j] * x0_dvc[p]; 	// < n calls to global memory
		j++;
	}

	// 2.- Output
	x_dvc[global_id] = a;	// particle weights
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
deviceFunction
void vectorUpdate_dvc::operator()(const uint64_t global_id) const {

	#pragma unroll
	for (uint16_t k = 0; k < ELEMENTS_AT_A_TIME; k++) {
		if ((global_id + k) >= totalLength) { return; }
		x_dvc[global_id + k] = x0_dvc[global_id + k] + scalar * directionVector[global_id + k];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint16_t ConjugateGradientEngine::execute(
	deviceUniquePtr<floatType>& inputOutputVector,
	const deviceUniquePtr<floatType>& targetVector,
	const deviceUniquePtr<int64_t>& matrixIndeces,
	const deviceUniquePtr<floatType>& matrixValues,
	const uint16_t maxNeighbors,
	const uint16_t maxIterations,
	const double interpolationTolerance
) {
	// Determine threads and blocks for the simulation
	const uint64_t vectorLength = targetVector.size_count();
	const uint16_t Threads = fminf(THREADS_P_BLK, vectorLength);
	const uint64_t Blocks = floor((double)(vectorLength - 1) / Threads) + 1;

	// These are for the update_vec function
	const uint16_t Threads_2 = fminf(THREADS_P_BLK, (float)vectorLength / ELEMENTS_AT_A_TIME);
	const uint64_t Blocks_2 = floor((double)(vectorLength / ELEMENTS_AT_A_TIME - 1) / Threads) + 1;

	// ------------------ AUXILIARIES FOR THE INTEPROLATION PROC. ------------------------------- //
	// Auxiliary values
	intType  k = 1;	// to control possible overflow of iterations
	bool flag = true;	// to stop the iterations
	const double squaredTolerance = interpolationTolerance * interpolationTolerance;

	// Initialize Conjugate gradient method ----------------------------------------------------
	// Compute A * X0
	try{
		gpu_device.launchKernel(Blocks, Threads, COOMatVecMultiplication_dvc{
			m_temp_dvc.get(),
			inputOutputVector.get(),
			matrixIndeces.get(),
			matrixValues.get(),
			vectorLength,
			maxNeighbors
		});
	}
	catch (const std::exception& except) {
		std::cerr << "Caught exception at matrix multiplication: " << except.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Compute R = B-A*X0
	gpu_device.launchKernel(Blocks_2, Threads_2, vectorUpdate_dvc{
		m_R_dvc.get(),
		targetVector.get(),
		(floatType) -1,
		m_temp_dvc.get()
	});

	floatType Alpha, R0_norm, r_squaredNorm, aux, beta;

	// Assign P = R
	gpu_device.memCpy_dvc2dvc(
		m_P_dvc.get(), m_R_dvc.get(), m_R_dvc.size_bytes()
	);

	while (flag) { // this flag is useful to know when we have arrived to the desired tolerance
		// Alpha computation (EVERYTHING IS CORRECT!)
			// 1.1.- Compute AP=A*P
		try{
			gpu_device.launchKernel(Blocks, Threads, COOMatVecMultiplication_dvc{
				m_AP_dvc.get(),
				m_P_dvc.get(),
				matrixIndeces.get(),
				matrixValues.get(),
				vectorLength,
				maxNeighbors
			});
		} 
		catch (const std::exception& except) {
			std::cerr << "Error caught in matrix multiplication at line: " << __LINE__ << ".\n";
			std::cerr << "Exception: " << except.what() << std::endl;
			return EXIT_FAILURE;
		}

		// 1.2.- Compute P'*AP
		aux = innerProduct_dvc<floatType>(m_P_dvc.get(), m_AP_dvc.get(), vectorLength);

		// 1.3.- R'*R
		R0_norm = innerProduct_dvc<floatType>(m_R_dvc.get(), m_R_dvc.get(), vectorLength);

		Alpha = R0_norm / aux;

		// New X and R: (new, old, scalar, driving vec, total length)
		// 1.- Update Lambdas
		try {
			gpu_device.launchKernel(Blocks_2, Threads_2, vectorUpdate_dvc{
				inputOutputVector.get(),
				inputOutputVector.get(),
				Alpha,
				m_P_dvc.get(),
				vectorLength
			});
		}
		catch (const std::exception& except) {
			std::cerr << "Error updating vector at line " << __LINE__ << ". ";
			std::cerr << "Exception: " << except.what() << std::endl;
			return EXIT_FAILURE;
		}

		// 2.- Update residuals 
		try {
			gpu_device.launchKernel(Blocks_2, Threads_2, vectorUpdate_dvc{
				m_R_dvc.get(),
				m_R_dvc.get(),
				-Alpha,
				m_AP_dvc.get(),
				vectorLength
				});
		}
		catch (const std::exception& except) {
			std::cerr << "Error updating vector at line " << __LINE__ << ". ";
			std::cerr << "Exception: " << except.what() << std::endl;
			return EXIT_FAILURE;
		}

		// Compute residual l_2 norm
		r_squaredNorm = innerProduct_dvc<floatType>(m_R_dvc.get(), m_R_dvc.get(), (floatType)0);

		if ((double)r_squaredNorm / (vectorLength * vectorLength) < squaredTolerance) {
			flag = false;
			break;
		}
		else if (k > maxIterations) {
			std::cout << "No convergence was obtained after reaching max. allowed iterations. Last residual norm was: " << sqrt(r_squaredNorm) << "\n";

			k = -1;
			flag = false;
			break;
		}
		else {
			beta = r_squaredNorm / R0_norm;

			try {
				gpu_device.launchKernel(Blocks_2, Threads_2, vectorUpdate_dvc{
					m_P_dvc.get(),
					m_R_dvc.get(),
					beta,
					m_P_dvc.get(),
					vectorLength
					});
			}
			catch (const std::exception& except) {
				std::cerr << "Error updating vector at line " << __LINE__ << ". ";
				std::cerr << "Exception: " << except.what() << std::endl;
				return EXIT_FAILURE;
			}
			k++;
		}
	}
	return k;	// In this case, we return the iteration number, contrarily to returning the success/failure of the function
}
