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
										const grid<DIM, T> 			BoundingBox,
										const grid<DIM, T> 			Base_Mesh,
										const T				rescaling){

	const uint64_t globalID = blockDim.x * blockIdx.x + threadIdx.x;

	if(globalID >= BoundingBox.Total_Nodes() / powf(2,DIM)) { return; }

	// This way we can obtain the global index of the cube vertex from the cube vertex's relative position
	INT main_node_idx = 0;

	for (uint16_t j = 0; j < DIM; j++) {
		main_node_idx += floor(positive_rem(globalID, pow(BoundingBox.Nodes_per_Dim / rescaling, j + 1)) / pow(BoundingBox.Nodes_per_Dim / rescaling, j)) * pow(BoundingBox.Nodes_per_Dim, j) * rescaling;
	}

	// find the dyadic cube indeces and do the wavelet transform at the same time
	for(uint16_t d = 0; d < DIM; d++){
		for(UINT k = 0; k < powf(2,DIM);k++){

			if( floor(positive_rem(k, pow(2, d + 1)) / pow(2, d))  == 0 ){ // this part decides whether it is an approximation node

				INT approx_coef_idx = main_node_idx;

				for (uint16_t j = 0; j < DIM; j++){
					approx_coef_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(BoundingBox.Nodes_per_Dim, j) * rescaling/2;
				}

				INT detail_coef_idx = approx_coef_idx + pow(BoundingBox.Nodes_per_Dim, d) * rescaling / 2;

				// Transform the indeces to the base mesh
				approx_coef_idx = Base_Mesh.Indx_here(approx_coef_idx, BoundingBox);
				detail_coef_idx = Base_Mesh.Indx_here(detail_coef_idx, BoundingBox);

				if(approx_coef_idx > -1 && approx_coef_idx < Base_Mesh.Total_Nodes() && detail_coef_idx > -1 && detail_coef_idx < Base_Mesh.Total_Nodes()){
					_1D_WVLET<T>(PDF[approx_coef_idx], PDF[detail_coef_idx]);
				}
			}
		}
	}

	// Now, we will recheck all the wavelet coefficients and compare with the threshold
	Activate_node[main_node_idx].node = Base_Mesh.Indx_here(main_node_idx, BoundingBox);

	for(UINT k = 1; k < powf(2,DIM); k++){

		INT temp_idx = main_node_idx;
		
		// PROBLEM HERE, PROBABLY
		for (uint16_t j = 0; j < DIM; j++){
			temp_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(BoundingBox.Nodes_per_Dim, j) * rescaling / 2;
		}

		INT temp_2 = Base_Mesh.Indx_here(temp_idx, BoundingBox);

		if (temp_2 > -1 && temp_2 < Base_Mesh.Total_Nodes()) {
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
								const grid<DIM, T>&				Base_Mesh,
								grid<DIM, T>&					Supp_BBox,
								const int						LvlFine, 
								const int						LvlCoarse) {
	// Final AMR procedure


	// 1st.- Tweak the Support Bounding Box so that the AMR is useful!
	UINT Eff_Finest_Level = (UINT) ceil(log2(Supp_BBox.Nodes_per_Dim));

	// If I'm going to need, at least, the same amount of points as in my base mesh...just use the base mesh!
	if (Eff_Finest_Level >= log2(Base_Mesh.Nodes_per_Dim)) {
		Supp_BBox = Base_Mesh;
	}
	else {
		// Here we have to add the "make bigger bounding box" part (we need 2^something points per dimension)
		Supp_BBox.Nodes_per_Dim = pow(2, Eff_Finest_Level);

		for (uint16_t d = 0; d < DIM; d++) {
			Supp_BBox.Boundary_sup.dim[d] = Supp_BBox.Boundary_inf.dim[d] + Base_Mesh.Discr_length() * (Supp_BBox.Nodes_per_Dim - 1);
		}
	}

	thrust::host_vector<AMR_node_select> 	H__Node_selection(Supp_BBox.Total_Nodes(), {0,0});
	thrust::device_vector<AMR_node_select>	D__Node_selection = H__Node_selection;

	UINT rescaling = 2;

	for (uint16_t k = 0; k < Eff_Finest_Level; k++) {

		//Possible optimization: DO SEVERAL BLOCKS PER THREAD!!!
		uint16_t Threads = fmin(THREADS_P_BLK, Supp_BBox.Total_Nodes()/pow(rescaling,DIM) );
		UINT	 Blocks	 = floor((Supp_BBox.Total_Nodes()/pow(rescaling, DIM) - 1) / Threads) + 1;

		D__Wavelet_Transform__F<DIM, T> <<<Blocks, Threads>>> (rpc(D__PDF,0), rpc(D__Node_selection,0), Supp_BBox, Base_Mesh, rescaling);
		gpuError_Check(cudaDeviceSynchronize());

		rescaling *= 2;	// our grid<DIM, T> will now have half the number of points
	}

	thrust::sort(thrust::device, D__Node_selection.begin(),D__Node_selection.end());
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

		AdaptGrid.at(k) = Base_Mesh.Get_node(temp_idx);
		AdaptPDF .at(k) = H_PDF [temp_idx];
	}

	return 0;
}
#endif