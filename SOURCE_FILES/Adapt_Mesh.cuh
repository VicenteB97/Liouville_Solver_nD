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
	
	// Get the cube approximation vertex
	gridPoint<DIM, T> cube_app_vrtx = BoundingBox.Boundary_inf;
	INT main_idx_at_BBox = 0;

	for (uint16_t d = 0; d < DIM; d++) {
		INT temp_idx = floor(positive_rem(globalID, pow(BoundingBox.Nodes_per_Dim / rescaling, d + 1)) / pow(BoundingBox.Nodes_per_Dim / rescaling, d)) * rescaling;
		main_idx_at_BBox += temp_idx * pow(BoundingBox.Nodes_per_Dim, d);

		cube_app_vrtx.dim[d] += temp_idx * BoundingBox.Discr_length();
	}

	if (!Problem_Domain.Contains_particle(cube_app_vrtx)) { return; }

	// You could even get the index at the base mesh and then use it for checking the wavelet transform coefficients
	INT main_idx_at_Base = 0;
	for (uint16_t d = 0; d < DIM; d++) {
		main_idx_at_Base += roundf((cube_app_vrtx.dim[d] - Problem_Domain.Boundary_inf.dim[d]) / Problem_Domain.Discr_length()) * pow(Problem_Domain.Nodes_per_Dim, d);
	}

	// find the dyadic cube indeces and do the wavelet transform at the same time
	for(uint16_t d = 0; d < DIM; d++){
		for(UINT k = 0; k < powf(2,DIM);k++){

			if( floor(positive_rem(k, pow(2, d + 1)) / pow(2, d))  == 0 ){ // this part decides whether it is an approximation node

				gridPoint<DIM, T> approx_node = cube_app_vrtx;
				INT approx_idx = main_idx_at_Base;

				for (uint16_t j = 0; j < DIM; j++){
					approx_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(Problem_Domain.Nodes_per_Dim, j) * rescaling/2;
				}

				INT detail_idx = approx_idx + pow(Problem_Domain.Nodes_per_Dim, d) * rescaling / 2;

				// Transform the indeces to the base mesh
				if (detail_idx > -1 && detail_idx < Problem_Domain.Total_Nodes()) {

					_1D_WVLET<T>(PDF[approx_idx], PDF[detail_idx]);
				}
			}
		}
	}

	// Now, we will recheck all the wavelet coefficients and compare with the threshold
	for(UINT k = 1; k < powf(2,DIM); k++){

		// This will allow us to enter into the auxiliary variables set for the wavelet transform
		INT temp_idx = main_idx_at_BBox, indx_at_Problem_Domain = main_idx_at_Base;
		gridPoint<DIM, T> temp_GN = cube_app_vrtx;
		
		for (uint16_t j = 0; j < DIM; j++){
			INT temp = floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * rescaling / 2;
			
			temp_GN.dim[j]			+= temp * Problem_Domain.Discr_length();

			temp_idx				+= temp * pow(BoundingBox.Nodes_per_Dim, j);
			indx_at_Problem_Domain	+= temp * pow(Problem_Domain.Nodes_per_Dim, j);
		}

		// Global index at the bounding box grid 
		if (Problem_Domain.Contains_particle(temp_GN)) {

			Activate_node[temp_idx].node = indx_at_Problem_Domain;

			if (abs(PDF[indx_at_Problem_Domain]) >= TOLERANCE_AMR) {
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
		Supp_BBox.Nodes_per_Dim = pow(2, roundf(log2(Supp_BBox.Nodes_per_Dim)));

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