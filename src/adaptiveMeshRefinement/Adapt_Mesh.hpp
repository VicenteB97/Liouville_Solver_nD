#ifndef __ADAPT_MESH_HPP__
#define __ADAPT_MESH_HPP__

#include "waveletTransform/waveletTransform.hpp"
#include "include/headers.hpp"
#include "mesh/Domain.hpp"

hostFunction
void setInitialParticles(
	const floatType* inputSignal_dvc,
	deviceUniquePtr<Particle>& outputActiveNodes_dvc,
	deviceUniquePtr<floatType>& outputActiveNodesValues_dvc,
	const cartesianMesh& signalBoundingBox,
	const cartesianMesh& signalDomain
);

hostFunction
void getDetailAboveThresholdNodes(
	waveletTransform& amrEngine, 
	deviceUniquePtr<Particle>& particle_locations, 
	deviceUniquePtr<floatType>& particle_values, 
	const cartesianMesh& signalDomain
);


class writeSignalInBoundingBox {
public:
	const floatType* input_signal;
	floatType* output_signal;
	const cartesianMesh signalDomain;
	const cartesianMesh signalBoundingBox;
	const uint64_t max_nodes;
public:
	deviceFunction void operator()(const uint64_t global_id) const;
};


template<uint16_t elementsProcessedPerThread>
class get_nodes_from_indeces {
public:
	Particle* outputNodes;
	floatType* outputNodesValues;
	const floatType* transformedSignalInBoundingBox;
	const cartesianMesh inputNodes;
	const int64_t* nodeIdx;
	const uintType elementNr;
public:
	deviceFunction void operator()(const uint64_t global_id) const {
		#pragma unroll
		for (uint16_t k = 0; k < elementsProcessedPerThread; k++) {

			const uintType myIdx = global_id * elementsProcessedPerThread + k;

			if (myIdx < elementNr) {
				const uint64_t myNodeIdx = nodeIdx[myIdx];

				outputNodes[myIdx] = inputNodes.get_node(myNodeIdx);
				outputNodesValues[myIdx] = transformedSignalInBoundingBox[myNodeIdx];
			}
		}
	};
};

#endif