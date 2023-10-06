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
										const grid<DIM, T> 	Base_Mesh,
										const T				rescaling){

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if(globalID >= BoundingBox.Total_Nodes() / powf(rescaling,DIM)) { return; }

	// This way we can obtain the global index of the cube vertex from the cube vertex's relative position
	INT main_node_idx = 0;

	// This part gives the global index (IN THE BOUNDING BOX) of the approximation node
	for (uint16_t j = 0; j < DIM; j++) {
		main_node_idx += floor(positive_rem(globalID, pow(BoundingBox.Nodes_per_Dim / rescaling, j + 1)) / pow(BoundingBox.Nodes_per_Dim / rescaling, j)) * pow(BoundingBox.Nodes_per_Dim, j) * rescaling;
	}

	if (!Problem_Domain.Contains_particle(BoundingBox.Get_node(main_node_idx))) { return; }

	INT main_idx_at_Base = Problem_Domain.Indx_here(main_node_idx, BoundingBox);

	// find the dyadic cube indeces and do the wavelet transform at the same time
	for(uint16_t d = 0; d < DIM; d++){
		for(UINT k = 0; k < powf(2,DIM);k++){

			if( floor(positive_rem(k, pow(2, d + 1)) / pow(2, d))  == 0 ){ // this part decides whether it is an approximation node

				INT approx_coef_idx = main_idx_at_Base;

				for (uint16_t j = 0; j < DIM; j++){
					approx_coef_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(Problem_Domain.Nodes_per_Dim, j) * rescaling/2;
				}

				INT detail_coef_idx = approx_coef_idx + pow(Problem_Domain.Nodes_per_Dim, d) * rescaling / 2;

				// Transform the indeces to the base mesh
				if (detail_coef_idx > -1 && detail_coef_idx < Problem_Domain.Total_Nodes()) {

					_1D_WVLET<T>(PDF[approx_coef_idx], PDF[detail_coef_idx]);
				}
			}
		}
	}

	// Now, we will recheck all the wavelet coefficients and compare with the threshold

	for(UINT k = 1; k < powf(2,DIM); k++){

		// This will allow us to enter into the auxiliary variables set for the wavelet transform
		INT temp_idx = main_node_idx;
		
		for (uint16_t j = 0; j < DIM; j++){

			temp_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(BoundingBox.Nodes_per_Dim, j) * rescaling / 2;
		}

		// Global index at the bounding box grid 
		if (Problem_Domain.Contains_particle(BoundingBox.Get_node(temp_idx))) {

			INT temp_2 = Problem_Domain.Indx_here(temp_idx, BoundingBox);
			Activate_node[temp_idx].node = temp_2;

			if (abs(PDF[temp_2]) >= TOLERANCE_AMR) {
				Activate_node[temp_idx].AMR_selected = 1;
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
	
	thrust::host_vector<AMR_node_select> 	H__Node_selection(Supp_BBox.Total_Nodes(), {0,0});
	thrust::device_vector<AMR_node_select>	D__Node_selection = H__Node_selection;

	UINT rescaling = 2;

	Supp_BBox.Squarify();	// Make it square
	Supp_BBox.Nodes_per_Dim = floor( (Supp_BBox.Boundary_sup.dim[0] - Supp_BBox.Boundary_inf.dim[0]) / Problem_Domain.Discr_length() );

	if (fmod(log2(Supp_BBox.Nodes_per_Dim), 1) != 0) {
		Supp_BBox.Nodes_per_Dim = pow(2, 1 + floor(log2(Supp_BBox.Nodes_per_Dim)));

		if (Supp_BBox.Nodes_per_Dim == Problem_Domain.Nodes_per_Dim) {
			Supp_BBox = Problem_Domain;
		}
		else{
			Supp_BBox.Boundary_inf = Base_Mesh.Get_node(Base_Mesh.Get_binIdx(Supp_BBox.Boundary_inf));
			for (uint16_t d = 0; d < DIM; d++) {
				Supp_BBox.Boundary_sup.dim[d] = Supp_BBox.Boundary_inf.dim[d] + (Supp_BBox.Nodes_per_Dim - 1) * Problem_Domain.Discr_length();
			}
		}
	}

	for (uint16_t k = 0; k < log2(Supp_BBox.Nodes_per_Dim); k++) {

		uint16_t Threads = fmin(THREADS_P_BLK, Supp_BBox.Total_Nodes()/pow(rescaling,DIM) );
		UINT	 Blocks	 = floor((Supp_BBox.Total_Nodes()/pow(rescaling, DIM) - 1) / Threads) + 1;

		D__Wavelet_Transform__F<DIM, T> <<<Blocks, Threads>>> (rpc(D__PDF,0), rpc(D__Node_selection,0), Supp_BBox, Problem_Domain, Base_Mesh, rescaling);
		gpuError_Check(cudaDeviceSynchronize());

		rescaling *= 2;	// our grid<DIM, T> will now have half the number of points
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