#ifndef __REMESHING_HPP__
#define __REMESHING_HPP__

#include "CSRBF.hpp"
#include "include/headers.hpp"
#include "probabilityDistributions/Probability.hpp"
#include "mesh/Domain.hpp"
#include "mesh/Particle.hpp"

template<typename _Ty>
class correctNegativeValues {
public:
	_Ty* densityFunction;
	uint64_t nrNodes;
public:
	deviceFunction void operator()(const uint64_t global_id) {
		const uint64_t i = global_id * ELEMENTS_AT_A_TIME;

		#pragma unroll
		for (uint16_t k = 0; k < ELEMENTS_AT_A_TIME; k++) {
			if (i + k < nrNodes) {
				densityFunction[i + k] = fmaxf(densityFunction[i + k], (floatType) 0);
			}
		}
	}
};

template<typename _parameterVectorType>
class remeshParticles {
public:
	Particle* particlePositions;
	floatType* densityFunction;
	floatType* interpolationLambdas;
	const _parameterVectorType* parameterWeights;
	const intType* nrSamplesPerParameter;
	const floatType searchRadius;
	const uint64_t nrParticles;
	const uintType blockSamples;
	const uintType offset;
	const cartesianMesh Domain;
	const cartesianMesh expandedDomain;

public:
	deviceFunction void operator()(const uint64_t global_id) {
		if (global_id >= nrParticles * blockSamples) { return; }

		uintType Current_sample = offset + floor((double)global_id / nrParticles);
		Param_vec<PARAM_SPACE_DIMENSIONS>	aux = Gather_Param_Vec<PARAM_SPACE_DIMENSIONS>(Current_sample, parameterWeights, n_Samples);

		floatType weighted_lambda = interpolationLambdas[global_id] * aux.Joint_PDF;

		Particle particle(particlePositions[global_id]);

		// Find the point in the lowest corner of the search box!
		Particle Lowest_node(expandedDomain.get_node(expandedDomain.getBinIdx(particle, -lround(DISC_RADIUS))));

		const uintType Neighbors_per_dim = 2 * lround(DISC_RADIUS) + 1;
		const uintType totalNeighborsToVisit = pow(Neighbors_per_dim, PHASE_SPACE_DIMENSIONS);
		const floatType domainDiscretization = Domain.discr_length();

		// Go through all the nodes where rewriting will be possible
		for (uint16_t k = 0; k < totalNeighborsToVisit; k++) {

			Particle visit_node(Lowest_node);

			// Get the node at that point
			uintType tempPowerAccumulate = 1;

			#pragma unroll
			for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
				uintType temp_idx = floor((double)positive_rem(k, Neighbors_per_dim * tempPowerAccumulate) / tempPowerAccumulate);

				visit_node.dim[d] += temp_idx * domainDiscretization;
				tempPowerAccumulate *= Neighbors_per_dim;
			}

			// If it is inside our problem mesh...
			if (Domain.containsParticle(visit_node)) {

				// Calculate normalized distance
				floatType dist = visit_node.distance(particle) / search_radius;

				// if it's inside the RBF support...
				if (dist <= 1) {

					dist = RBF(search_radius, dist) * weighted_lambda;

					intType idx = Domain.getBinIdx(visit_node);

					atomicAdd(&densityFunction[idx], dist);
				}
			}
		}
	}
};

template<typename _Ty>
void L1normalizeLambdas(_Ty* lambdasFromInterpolation, const uint32_t normalizingValue){
#ifdef USECUDA
	thrust::device_ptr<_Ty> lambdasPointer(lambdasFromInterpolation);

	floatType temp = thrust::reduce(thrust::device, lambdasPointer.begin(), lambdasPointer.end());
	thrust::transform(lambdasPointer.begin(), lambdasPointer.end(), lambdasPointer.begin(), normalizingValue / temp * _1);
#endif
};
#endif