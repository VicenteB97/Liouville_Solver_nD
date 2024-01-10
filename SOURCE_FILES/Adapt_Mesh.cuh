#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "Constants.cuh"
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
__global__ void D__Wavelet_Transform__F(	TYPE*	PDF,
								//   AMR_node_select* 	Activate_node,
								  		UINT* nodeIdxs,
										UINT* isActiveNode,
										const Mesh 	BoundingBox,
										const Mesh	Problem_Domain,
										const TYPE	rescaling){

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// Range guard for out-of-bounds nodes
	if (globalID >= BoundingBox.Total_Nodes() / powf(rescaling, PHASE_SPACE_DIMENSIONS)) { return; }

	// Global index of the main approximation vertex at the bounding box
	INT cube_app_IDX = 0;

	// Compute the index and the node per se
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		INT temp_idx = floor(positive_rem(globalID, pow(BoundingBox.Nodes_per_Dim / rescaling, d + 1)) / pow(BoundingBox.Nodes_per_Dim / rescaling, d)) * rescaling;

		cube_app_IDX += temp_idx * pow(BoundingBox.Nodes_per_Dim, d);
	}

	// 1 set of wavelets per dimension (1D: horizontal; 2D: Horizontal + Vertical; 3D: Horz + Vert + Deep; ...)
	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {

		// Go through all the vertices that are defined by the main cube approximation vertex
		for (INT k = 0; k < pow(2, PHASE_SPACE_DIMENSIONS); k++) {

			// If we are at the current approximation vertex:
			if (floor(positive_rem(k, pow(2, d + 1)) / pow(2, d)) == 0) {

				// Copmute approximation node
				INT app_IDX_at_BBox = cube_app_IDX;

				for (uint16_t j = 0; j < PHASE_SPACE_DIMENSIONS; j++) {
					INT temp = floorf(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * rescaling / 2;	// j-th component index

					app_IDX_at_BBox += temp * pow(BoundingBox.Nodes_per_Dim, j);
				}

				// Compute corresponding detail node
				INT det_IDX_at_BBox = app_IDX_at_BBox + pow(BoundingBox.Nodes_per_Dim, d) * rescaling / 2;

				Particle app_node = BoundingBox.Get_node(app_IDX_at_BBox);
				Particle det_node = BoundingBox.Get_node(det_IDX_at_BBox);

				// Check which ones are in the problem domain!
				if (Problem_Domain.Contains_particle(app_node) && Problem_Domain.Contains_particle(det_node)) {

					// Calculate the indeces for the problem domain
					INT app_node_at_PD = Problem_Domain.Get_binIdx(app_node);
					INT det_node_at_PD = Problem_Domain.Get_binIdx(det_node);

					_1D_WVLET(PDF[app_node_at_PD], PDF[det_node_at_PD]);
				}
			}
		}
	}

	// Now we have to go see what happens with the outputs
	nodeIdxs[cube_app_IDX] = 0;

	for (UINT k = 1; k < pow(2, PHASE_SPACE_DIMENSIONS); k++) {

		Particle visit_node = BoundingBox.Get_node(cube_app_IDX);

		// Get the indeces at the bounding box:
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			INT temp = floorf(positive_rem(k, pow(2, d + 1)) / pow(2, d)) * rescaling / 2;	// j-th component index

			visit_node.dim[d] += temp * BoundingBox.Discr_length();
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

		if(myIdx >= elementNr){ return; }

		const INT myNodeIdx = nodeIdx[myIdx];

		outputPDF[myIdx] = inputPDF[myNodeIdx];
		outputNodes[myIdx] = inputNodes.Get_node(myNodeIdx);

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
							// std::vector<TYPE>&					AdaptPDF, 
							// std::vector<Particle>& 				AdaptGrid,
							const Mesh&							Problem_Domain,
							const Mesh&							Base_Mesh,
							Mesh&								Supp_BBox) {


	UINT rescaling = 2;

	// Prepare the bounding box for the Wavelet-based AMR procedure! (We don't really care if it's off limits from the problem domain)
	Supp_BBox.Squarify();	// Make it square
	Supp_BBox.Nodes_per_Dim = (Supp_BBox.Boundary_sup.dim[0] - Supp_BBox.Boundary_inf.dim[0]) / Problem_Domain.Discr_length() + 1;

	const double refinementLvl = log2(Supp_BBox.Nodes_per_Dim);

	if (fmod(refinementLvl, 1) != 0) {
		Supp_BBox.Nodes_per_Dim = pow(2, ceil(refinementLvl));

		if (Supp_BBox.Nodes_per_Dim >= Problem_Domain.Nodes_per_Dim) {Supp_BBox = Problem_Domain;}
		else{
			Supp_BBox.Boundary_inf = Base_Mesh.Get_node(Base_Mesh.Get_binIdx(Supp_BBox.Boundary_inf));	// To make sure it falls into the mesh nodes

			#pragma unroll
			for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
				Supp_BBox.Boundary_sup.dim[d] = Supp_BBox.Boundary_inf.dim[d] + (Supp_BBox.Nodes_per_Dim - 1) * Problem_Domain.Discr_length();
			}
		}
	}

	// thrust::host_vector<AMR_node_select> 	H__Node_selection(Supp_BBox.Total_Nodes(), { 0,0 });
	// thrust::device_vector<AMR_node_select>	D__Node_selection = H__Node_selection;

	thrust::device_vector<UINT> nodeIdxs(Supp_BBox.Total_Nodes(), 0);
	thrust::device_vector<UINT> isAssignedNode(Supp_BBox.Total_Nodes(), 0);

	for (uint16_t k = 0; k < log2(Supp_BBox.Nodes_per_Dim); k++) {

		uint16_t Threads = fmin(THREADS_P_BLK, Supp_BBox.Total_Nodes()/pow(rescaling,PHASE_SPACE_DIMENSIONS) );
		UINT	 Blocks	 = floor((Supp_BBox.Total_Nodes()/pow(rescaling, PHASE_SPACE_DIMENSIONS) - 1) / Threads) + 1;

		D__Wavelet_Transform__F <<<Blocks, Threads>>> (rpc(D__PDF,0), rpc(nodeIdxs,0), rpc(isAssignedNode, 0), Supp_BBox, Problem_Domain, rescaling);
		gpuError_Check(cudaDeviceSynchronize());

		rescaling *= 2;	// our Mesh will now have half the number of points
	}
	
	// Get the number of assigned nodes
	const UINT nrSelectedNodes = thrust::reduce(thrust::device, isAssignedNode.begin(), isAssignedNode.end());
	
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

	return 0;
}


#endif