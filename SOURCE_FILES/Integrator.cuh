#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

#include "Constants.cuh"
#include "Probability.cuh"
#include "Domain.cuh"

using namespace thrust::placeholders; // this is useful for the multiplication of a device vector by a constant

/// @brief This function computes the advection of the particles created by AMR.
/// 	It uses the RK4 scheme for the spatial variables advection and the Simpson rule for the exponential integral.
/// @param Particles Particle location (spatial variables)
/// @param PDF PDF value at the corresponding particle location
/// @param parameters Parameters to be used for the vector field and its corresponding divergence function
/// @param t0 Inital time for starting the simulation
/// @param deltaT Time step used in the simulation
/// @param ReinitSteps Number of steps before needing a re-interpolation
/// @param Adapt_Points Number of particles as computed by the AMR scheme
/// @param Random_Samples Number of random paramRealization samples
/// @return 
__global__ void ODE_INTEGRATE(gridPoint* Particles,
							TYPE* PDF,
							const TYPE* 			 lambdas,
							const Param_pair* parameters,
							const INT* n_Samples,
							double				t0,
							const double			deltaT,
							const double		tF,
							const INT		Adapt_Points,
							const INT		Random_Samples,
							const UINT		mode,
							const double* extra_param,
							const grid D_Mesh) {

	const uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < Adapt_Points * Random_Samples) {

		// AUXILIARY DATA TO RUN THE ITERATIONS
		// So, the total amount of advections are going to be: (no. particles x no. of samples)
		const UINT  i_sample = floor((double)i / Adapt_Points);
		const UINT  i_particle = positive_rem(i, Adapt_Points);

		const Param_vec<PARAM_SPACE_DIMENSIONS> paramRealization = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(i_sample, parameters, n_Samples);

		gridPoint k0, k1, k2, k3, k_final, temp, x0 = Particles[i];	// register storing the initial particle loc. ;
		TYPE	  Int1, Int2, Int3, Int_PDF = lambdas[i_particle];	// register storing the initial particle value;

		while (t0 < tF - deltaT / 2) {
			// Particle flow
			k0 = VECTOR_FIELD(x0, t0, paramRealization, mode, extra_param);

			temp = k0.Mult_by_Scalar(deltaT / 2.0f);
			k1 = VECTOR_FIELD(x0 + temp, t0 + deltaT / 2.0f, paramRealization, mode, extra_param);

			temp = k1.Mult_by_Scalar(deltaT / 2.0f);
			k2 = VECTOR_FIELD(x0 + temp, t0 + deltaT / 2.0f, paramRealization, mode, extra_param);

			temp = k2.Mult_by_Scalar(deltaT);
			k3 = VECTOR_FIELD(x0 + temp, t0 + deltaT, paramRealization, mode, extra_param);

			k1 = k1.Mult_by_Scalar(2.00f);
			k2 = k2.Mult_by_Scalar(2.00f);

			temp = k0 + k3 + k1 + k2;
			temp = x0 + temp.Mult_by_Scalar((TYPE)deltaT / 6.00f); // New particle dim

			// Integration of PDF: The following line corresponds to computing the approximation via a Hermite interpolation (we know initial and final points and their velocities)
			Int1 = DIVERGENCE_FIELD(x0, t0, paramRealization, mode, extra_param);

			k_final = VECTOR_FIELD(temp, t0 + deltaT, paramRealization, mode, extra_param);
			x0 = (x0 + temp).Mult_by_Scalar(0.50f);
			x0 = x0 + (k0 + k_final).Mult_by_Scalar(0.125f);
			Int2 = DIVERGENCE_FIELD(x0, (TYPE)(2.00f * t0 + deltaT) / 2.00f, paramRealization, mode, extra_param);

			Int3 = DIVERGENCE_FIELD(temp, (TYPE)t0 + deltaT, paramRealization, mode, extra_param);

			Int_PDF *= expf((float) -deltaT / 6.00f * (Int1 + 4.00f * Int2 + Int3)); // New particle value

			// Reinit step
			x0 = temp;
			t0 += deltaT;

			if (!D_Mesh.Contains_particle(temp)) { Int_PDF = 0; break; }	// This condition is equivalent to the homogeneous Neumann condition
		}

		Particles[i] = temp;
		PDF[i] = Int_PDF;
	}
}
#endif