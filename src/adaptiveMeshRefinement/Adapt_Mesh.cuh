#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "waveletTransform/waveletTransform.hpp"
#include "include/headers.hpp"
#include "mesh/Domain.hpp"

hostFunction
int32_t setInitialParticles(
	const floatType* input_signal_dvc,
	Particle* output_active_nodes_dvc,
	const cartesianMesh& signal_bounding_box,
	const cartesianMesh& signal_domain
);

hostFunction
void get_detail_above_threshold_nodes(waveletTransform& amr_engine, Particle* particle_locations, const cartesianMesh& signal_domain);


class write_signal_in_bounding_box {
public:
	const floatType* input_signal;
	floatType* output_signal;
	const cartesianMesh signal_domain;
	const cartesianMesh signal_bounding_box;
	const uint64_t max_nodes;
public:
	deviceFunction void operator()(const uint64_t global_id) const;
};


template<uint16_t elementsProcessedPerThread>
class get_nodes_from_indeces {
public:
	Particle* outputNodes;
	const cartesianMesh inputNodes;
	const uint64_t* nodeIdx;
	const uint32_t elementNr;
public:
	deviceFunction void operator()(const uint64_t global_id) const {
		#pragma unroll
		for (uint16_t k = 0; k < elementsProcessedPerThread; k++) {

			const uintType myIdx = global_id * elementsProcessedPerThread + k;

			if (myIdx < elementNr) {
				const uint64_t myNodeIdx = nodeIdx[myIdx];
				outputNodes[myIdx] = inputNodes.get_node(myNodeIdx);
			}
		}
	};
};

#endif