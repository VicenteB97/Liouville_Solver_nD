#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "headers.cuh"
#include "utils/error_functs.cuh"
#include "utils/numeric_defs.cuh"
#include "utils/others.cuh"
#include "Case_definition.cuh"
#include "Domain.cuh"
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief (DEVICE FUNCTION) Compute a 1D Haar wavelet transform
/// @param s1 
/// @param s2 
/// @return 
__device__ inline void _1D_WVLET(floatType& s1, floatType& s2);

/// @brief (GLOBAL FUNCTION) Compute 1 level of the multidimensional wavelet transform in the GPU
/// @tparam PHASE_SPACE_DIMENSIONS
/// @tparam floatType
/// @param PDF Our "signal". The multidimensional signal we want to compress
/// @param Activate_node An array with the nodes and the indication whether the node is chosen or not
/// @param BoundingBox The "smallest" cartesianMesh where the support of the PDF is contained
/// @param Problem_Domain Problem domain
/// @param rescaling Rescaling value that indicates the level of the wavelet transform
/// @return 
__global__ void D__Wavelet_Transform__F(
	floatType* PDF,
	uintType* nodeIdxs,
	uintType* isActiveNode,
	const cartesianMesh 	BoundingBox,
	const cartesianMesh	Problem_Domain,
	const floatType	rescaling);


template<uint16_t elementsProcessedPerThread>
__global__ void customAssignToGpuArray(
	floatType* outputPDF, 
	const floatType* inputPDF, 
	Particle* outputNodes, 
	const cartesianMesh inputNodes,
	const uintType* nodeIdx, 
	const intType elementNr);

/// @brief (HOST FUNCTION)
/// @param H_PDF 
/// @param D__PDF 
/// @param AdaptPDF 
/// @param AdaptGrid 
/// @param Problem_Domain 
/// @param Base_cartesianMesh 
/// @param Supp_BBox 
/// @return Error code (0 = good, -1 = something went wrong)
int16_t setInitialParticles(
	const thrust::host_vector<floatType>& H_PDF,
	thrust::device_vector<floatType>& D__PDF,
	thrust::device_vector<floatType>& AdaptPDF,
	thrust::device_vector<Particle>& AdaptGrid,
	const cartesianMesh& Problem_Domain,
	const cartesianMesh& Base_cartesianMesh,
	cartesianMesh& Supp_BBox);

#endif