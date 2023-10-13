//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
// This part contains the definitions of the functions regarding the whole AMR procedure					   //
// We STRONGLY RECOMMEND not changing absolutely anything if your only use of this program is to simulate	   //
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------//


#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "Classes.cuh"

// ---------------------- FUNCTIONS FOR General Dimension AMR ---------------------- //
template<class T>
__host__ __device__ 
inline void _1D_WVLET(T& s1, T& s2){

	T aux = 0.5*(s1 + s2);
	s2 	= s1 - s2;
	s1 	= aux;
}

template<uint16_t DIM, class T>
__global__ void D__Wavelet_Transform__F(T* 					PDF,
										AMR_node_select* 	Activate_node,
										const grid<DIM, T> 	BoundingBox,
										const grid<DIM, T> 	Problem_Domain,
										const T				rescaling){

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	// Range guard for out-of-bounds nodes
	if (globalID >= BoundingBox.Total_Nodes() / powf(rescaling, DIM)) { return; }

	// Global index of the main approximation vertex at the bounding box
	INT cube_app_IDX = 0;

	// Compute the index and the node per se
	for (uint16_t d = 0; d < DIM; d++) {
		INT temp_idx = floor(positive_rem(globalID, pow(BoundingBox.Nodes_per_Dim / rescaling, d + 1)) / pow(BoundingBox.Nodes_per_Dim / rescaling, d)) * rescaling;

		cube_app_IDX += temp_idx * powf(BoundingBox.Nodes_per_Dim, d);
	}

	// 1 set of wavelets per dimension (1D: horizontal; 2D: Horizontal + Vertical; 3D: Horz + Vert + Deep; ...)
	for (uint16_t d = 0; d < DIM; d++) {

		// Go through all the vertices that are defined by the main cube approximation vertex
		for (INT k = 0; k < powf(2, DIM); k++) {

			// If we are at the current approximation vertex:
			if (floor(positive_rem(k, pow(2, d + 1)) / pow(2, d)) == 0) {

				// Copmute approximation node
				INT app_IDX_at_BBox = cube_app_IDX;

				for (uint16_t j = 0; j < DIM; j++) {
					INT temp = floorf(positive_rem(k, powf(2, j + 1)) / powf(2, j)) * rescaling / 2;	// j-th component index

					app_IDX_at_BBox += temp * powf(BoundingBox.Nodes_per_Dim, j);
				}

				// Compute corresponding detail node
				INT det_IDX_at_BBox = app_IDX_at_BBox + powf(BoundingBox.Nodes_per_Dim, d) * rescaling / 2;

				gridPoint<DIM, T> app_node(BoundingBox.Get_node(app_IDX_at_BBox));
				gridPoint<DIM, T> det_node(BoundingBox.Get_node(det_IDX_at_BBox));

				// Check which ones are in the problem domain!
				if (Problem_Domain.Contains_particle(app_node) && Problem_Domain.Contains_particle(det_node)) {

					// Calculate the indeces for the problem domain
					INT app_node_at_PD = Problem_Domain.Get_binIdx(app_node);
					INT det_node_at_PD = Problem_Domain.Get_binIdx(det_node);

					_1D_WVLET<T>(PDF[app_node_at_PD], PDF[det_node_at_PD]);
				}
			}
		}
	}

	// Now we have to go see what happens with the outputs
	Activate_node[cube_app_IDX].node = 0;

	for (UINT k = 1; k < powf(2, DIM); k++) {

		INT temp_IDX_at_BBox = cube_app_IDX;

		// Get the indeces at the bounding box:
		for (uint16_t d = 0; d < DIM; d++) {
			INT temp = floorf(positive_rem(k, powf(2, d + 1)) / powf(2, d)) * rescaling / 2;	// j-th component index

			temp_IDX_at_BBox += temp * powf(BoundingBox.Nodes_per_Dim, d);
		}

		gridPoint<DIM, T> temp_node(BoundingBox.Get_node(temp_IDX_at_BBox));

		if (Problem_Domain.Contains_particle(temp_node)) {

			INT temp = Problem_Domain.Get_binIdx(temp_node);

			Activate_node[temp_IDX_at_BBox].node = temp;

			if (abs(PDF[temp]) >= TOLERANCE_AMR) {
				Activate_node[temp_IDX_at_BBox].AMR_selected = 1;
			}
		}
	}
}

template<uint16_t DIM, class T>
int16_t ADAPT_MESH_REFINEMENT_nD(const thrust::host_vector<T>&	H_PDF, 
								thrust::device_vector<T>&		D__PDF, 
								std::vector<T>&					AdaptPDF, 
								std::vector<gridPoint<DIM, T>>& AdaptGrid,
								const grid<DIM, T>&				Problem_Domain,
								const grid<DIM, T>&				Base_Mesh,
								grid<DIM, T>&					Supp_BBox) {


	UINT rescaling = 2;

	Supp_BBox.Squarify();	// Make it square
	Supp_BBox.Nodes_per_Dim = (Supp_BBox.Boundary_sup.dim[0] - Supp_BBox.Boundary_inf.dim[0]) / Problem_Domain.Discr_length() + 1;

	if (fmod(log2(Supp_BBox.Nodes_per_Dim), 1) != 0) {
		Supp_BBox.Nodes_per_Dim = pow(2, ceilf(log2(Supp_BBox.Nodes_per_Dim)));

		if (Supp_BBox.Nodes_per_Dim == Problem_Domain.Nodes_per_Dim) {
			Supp_BBox = Problem_Domain;
		}
		else{
			for (uint16_t d = 0; d < DIM; d++) {
				Supp_BBox.Boundary_sup.dim[d] = Supp_BBox.Boundary_inf.dim[d] + (Supp_BBox.Nodes_per_Dim - 1) * Problem_Domain.Discr_length();
			}
		}
	}

	// FIX PROBLEM WHEN THIS HAPPENS! EVERYTHING WORKS FINE IN 2D...3D IS GIVING PROBLEMS
	Supp_BBox = Problem_Domain;

	thrust::host_vector<AMR_node_select> 	H__Node_selection(Supp_BBox.Total_Nodes(), { 0,0 });
	thrust::device_vector<AMR_node_select>	D__Node_selection = H__Node_selection;

	for (uint16_t k = 0; k < log2(Supp_BBox.Nodes_per_Dim); k++) {

		uint16_t Threads = fmin(THREADS_P_BLK, Supp_BBox.Total_Nodes()/pow(rescaling,DIM) );
		UINT	 Blocks	 = floor((Supp_BBox.Total_Nodes()/pow(rescaling, DIM) - 1) / Threads) + 1;

		D__Wavelet_Transform__F<DIM, T> <<<Blocks, Threads>>> (rpc(D__PDF,0), rpc(D__Node_selection,0), Supp_BBox, Problem_Domain, rescaling);
		gpuError_Check(cudaDeviceSynchronize());

		rescaling *= 2;	// our grid will now have half the number of points
	}

	thrust::sort(thrust::device, D__Node_selection.begin(), D__Node_selection.end());
	H__Node_selection = D__Node_selection;

	UINT counter = 0;
	while(H__Node_selection[counter].AMR_selected == 1){
		counter++;
	}

	AdaptGrid.resize(counter);
	AdaptPDF .resize(counter);

#pragma omp parallel for
	for(INT k = 0; k < counter; k++){
		INT temp_idx = H__Node_selection[k].node;

		AdaptGrid.at(k) = Problem_Domain.Get_node(temp_idx);
		AdaptPDF .at(k) = H_PDF [temp_idx];
	}

	return 0;
}
#endif