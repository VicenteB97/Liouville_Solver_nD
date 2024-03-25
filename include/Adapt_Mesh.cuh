#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "headers.cuh"
#include "utils/error_functs.cuh"
#include "utils/numeric_defs.cuh"
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
__device__ inline void _1D_WVLET(TYPE& s1, TYPE& s2){

	TYPE aux = 0.5*(s1 + s2);
	s2 	= s1 - s2;
	s1 	= aux;
}

/// @brief (GLOBAL FUNCTION) Compute 1 level of the multidimensional wavelet transform in the GPU
/// @tparam PHASE_SPACE_DIMENSIONS
/// @tparam TYPE
/// @param PDF Our "signal". The multidimensional signal we want to compress
/// @param Activate_node An array with the nodes and the indication whether the node is chosen or not
/// @param BoundingBox The "smallest" Mesh where the support of the PDF is contained
/// @param Problem_Domain Problem domain
/// @param rescaling Rescaling value that indicates the level of the wavelet transform
/// @return 
__global__ void D__Wavelet_Transform__F(TYPE*		PDF,
								  		UINT*		nodeIdxs,
										UINT*		isActiveNode,
										const Mesh 	BoundingBox,
										const Mesh	Problem_Domain,
										const TYPE	rescaling){

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// Range guard for out-of-bounds nodes
	if (globalID >= BoundingBox.Total_Nodes() / powf(rescaling, PHASE_SPACE_DIMENSIONS)) { return; }

	const UINT		totalNodes		= Problem_Domain.Total_Nodes();		// Total nodes in the problem domain
	const uint16_t	miniSquareNodes = pow(2, PHASE_SPACE_DIMENSIONS);	// Total nodes in each simple wavelet transform (per GPU thread)

	// Global index of the main approximation vertex at the bounding box
	INT cube_app_IDX = 0;

	// Compute the index and the node per se

	UINT multCounter = 1;	// auxiliary counter: => pow(BoundingBox.Nodes_per_Dim / rescaling, d)
	UINT multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		INT temp_idx = floorf(positive_rem(globalID, multCounter * (BoundingBox.Nodes_per_Dim / rescaling)) / multCounter) * rescaling;

		cube_app_IDX += temp_idx * multCounter_2;
		multCounter *= BoundingBox.Nodes_per_Dim / rescaling;
		multCounter_2 *= BoundingBox.Nodes_per_Dim;
	}

	multCounter = 1;	// Reinitialize for next computations: => pow(2, d)
	multCounter_2 = 1;	// For the BoundingBox: => pow(BoundingBox.Nodes_per_Dim, d)

	// 1 set of wavelets per dimension (1D: horizontal; 2D: Horizontal + Vertical; 3D: Horz + Vert + Deep; ...)
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {

		// Go through all the vertices that are defined by the main cube approximation vertex
		for (UINT k = 0; k < miniSquareNodes; k++) {

			// If we are at the current approximation vertex:
			if (floorf(positive_rem(k, 2 * multCounter) / multCounter) == 0) {		// here, multCounter == pow(2, d)

				// Copmute approximation node
				UINT app_IDX_at_BBox = cube_app_IDX;

				UINT multCounter_3 = 1;	// => pow(2, j)
				UINT multCounter_4 = 1;	// => pow(BoundingBox.Nodes_per_Dim, j)

				for (uint16_t j = 0; j < PHASE_SPACE_DIMENSIONS; j++) {
					INT temp = floorf(positive_rem(k, multCounter_3 * 2) / multCounter_3) * rescaling / 2;	// j-th component index

					app_IDX_at_BBox += temp * multCounter_4;
					multCounter_3	*= 2;
					multCounter_4	*= BoundingBox.Nodes_per_Dim;
				}

				// Compute corresponding detail node
				INT det_IDX_at_BBox = app_IDX_at_BBox + multCounter_2 * rescaling / 2;

				Particle app_node(BoundingBox.Get_node(app_IDX_at_BBox));
				Particle det_node(BoundingBox.Get_node(det_IDX_at_BBox));

				// Check which ones are in the problem domain!
				if (Problem_Domain.Contains_particle(app_node) && Problem_Domain.Contains_particle(det_node)) {

					// Calculate the indeces for the problem domain
					INT app_node_at_PD(Problem_Domain.Get_binIdx(app_node));
					INT det_node_at_PD(Problem_Domain.Get_binIdx(det_node));

					_1D_WVLET(PDF[app_node_at_PD], PDF[det_node_at_PD]);
				}
			}
		}
		multCounter		*= 2;
		multCounter_2	*= BoundingBox.Nodes_per_Dim;
	}

	// Now we have to go see what happens with the outputs
	nodeIdxs[cube_app_IDX] = 0;

	for (UINT k = 1; k < miniSquareNodes; k++) {

		Particle visit_node(BoundingBox.Get_node(cube_app_IDX));

		multCounter = 1;	// restart the counter

		// Get the indeces at the bounding box:
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT temp = floorf(positive_rem(k, multCounter * 2) / multCounter) * rescaling / 2;	// j-th component index

			visit_node.dim[d] += temp * BoundingBox.Discr_length();
			multCounter *= 2;
		}

		if (Problem_Domain.Contains_particle(visit_node)) {
			// These two indeces can be obtained in the previous loop
			INT temp = Problem_Domain.Get_binIdx(visit_node);
			INT temp_IDX_at_BBox = BoundingBox.Get_binIdx(visit_node);

			nodeIdxs[temp_IDX_at_BBox] = temp;

			if (abs(PDF[temp]) >= TOLERANCE_AMR) {
				isActiveNode[temp_IDX_at_BBox] = 1;
			}
		}
	}
}



template<uint16_t elementsProcessedPerThread>
__global__ void customAssignToGpuArray(TYPE* outputPDF, const TYPE* inputPDF, Particle* outputNodes, const Mesh inputNodes,
										 const UINT* nodeIdx, const INT elementNr){
	const int64_t globalId = blockDim.x * blockIdx.x + threadIdx.x;

	#pragma unroll
	for(uint16_t k = 0; k < elementsProcessedPerThread; k++){

		const UINT myIdx = globalId * elementsProcessedPerThread + k;

		if(myIdx < elementNr){
			const INT myNodeIdx = nodeIdx[myIdx];

			outputPDF[myIdx] = inputPDF[myNodeIdx];
			outputNodes[myIdx] = inputNodes.Get_node(myNodeIdx);
		}
	}
}

/// @brief (HOST FUNCTION)
/// @param H_PDF 
/// @param D__PDF 
/// @param AdaptPDF 
/// @param AdaptGrid 
/// @param Problem_Domain 
/// @param Base_Mesh 
/// @param Supp_BBox 
/// @return Error code (0 = good, -1 = something went wrong)
int16_t setInitialParticles(const thrust::host_vector<TYPE>&	H_PDF, 
							thrust::device_vector<TYPE>&		D__PDF, 
							thrust::device_vector<TYPE>&		AdaptPDF, 
							thrust::device_vector<Particle>& 	AdaptGrid,
							const Mesh&							Problem_Domain,
							const Mesh&							Base_Mesh,
							Mesh&								Supp_BBox) {


	UINT rescaling = 2;

	// Prepare the bounding box for the Wavelet-based AMR procedure! (We don't really care if it's off limits from the problem domain)
	Supp_BBox.Squarify();	// Make it square
	Supp_BBox.Nodes_per_Dim = (Supp_BBox.Boundary_sup.dim[0] - Supp_BBox.Boundary_inf.dim[0]) / Problem_Domain.Discr_length() + 1;

	double refinementLvl = log2(Supp_BBox.Nodes_per_Dim);

	if (fmod(refinementLvl, 1) != 0) {
		Supp_BBox.Nodes_per_Dim = pow(2, ceil(refinementLvl));

		// Rewrite the refinement level value
		refinementLvl = log2(Supp_BBox.Nodes_per_Dim);

		if (Supp_BBox.Nodes_per_Dim >= Problem_Domain.Nodes_per_Dim) {Supp_BBox = Problem_Domain;}
		else{
			Supp_BBox.Boundary_inf = Base_Mesh.Get_node(Base_Mesh.Get_binIdx(Supp_BBox.Boundary_inf));	// To make sure it falls into the mesh nodes

			#pragma unroll
			for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
				Supp_BBox.Boundary_sup.dim[d] = Supp_BBox.Boundary_inf.dim[d] + (Supp_BBox.Nodes_per_Dim - 1) * Problem_Domain.Discr_length();
			}
		}
	}

	thrust::device_vector<UINT> nodeIdxs(Supp_BBox.Total_Nodes(), 0);
	thrust::device_vector<UINT> isAssignedNode(Supp_BBox.Total_Nodes(), 0);

	for (uint16_t k = 0; k < refinementLvl; k++) {

		uint16_t Threads = fmin(THREADS_P_BLK, Supp_BBox.Total_Nodes()/pow(rescaling,PHASE_SPACE_DIMENSIONS) );
		UINT	 Blocks	 = floor((Supp_BBox.Total_Nodes()/pow(rescaling, PHASE_SPACE_DIMENSIONS) - 1) / Threads) + 1;

		D__Wavelet_Transform__F <<<Blocks, Threads>>> (rpc(D__PDF,0), rpc(nodeIdxs,0), rpc(isAssignedNode, 0), Supp_BBox, Problem_Domain, rescaling);
		gpuError_Check(cudaDeviceSynchronize());

		rescaling *= 2;	// our Mesh will now have half the number of points
	}
	
	// Get the number of assigned nodes
	const UINT nrSelectedNodes = thrust::reduce(thrust::device, isAssignedNode.begin(), isAssignedNode.end());
	if (nrSelectedNodes == 0) { std::cout << "\nError: AMR-selected points is 0...cannot continue\n"; return -1; }
	
	// Set the selected nodes first
	thrust::sort_by_key(thrust::device, isAssignedNode.begin(), isAssignedNode.end(), nodeIdxs.begin(), thrust::greater<INT>());

	// Reinitialize values to the PDF (we'll need it)
	D__PDF = H_PDF;

	AdaptGrid.resize(nrSelectedNodes); AdaptPDF.resize(nrSelectedNodes);

	const INT Threads = fmin(THREADS_P_BLK, nrSelectedNodes);
	const INT Blocks = floor((nrSelectedNodes - 1) / Threads) + 1;

	const uint16_t elementsAtATime = ELEMENTS_AT_A_TIME;

	customAssignToGpuArray<elementsAtATime><<<Threads,Blocks>>> (rpc(AdaptPDF,0), rpc(D__PDF,0), rpc(AdaptGrid,0), 
																	Problem_Domain, rpc(nodeIdxs, 0), nrSelectedNodes);
	gpuError_Check(cudaDeviceSynchronize());
	return 0;
}

#endif