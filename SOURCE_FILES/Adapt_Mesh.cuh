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
__host__ __device__ inline void _1D_WVLET(T& s1, T& s2){

	T aux = 0.5*(s1 + s2);
	s2 	= s1 - s2;
	s1 	= aux;
}

template<class T>
__global__ void D__Wavelet_Transform__F(T* 					PDF,
										AMR_node_select* 	Activate_node,
										const uint32_t 	PtsPerDim,
										const uint32_t 	Points_at_level,
										const uint32_t 	rescaling){

	const uint64_t globalID = blockDim.x*blockIdx.x + threadIdx.x;

	if(globalID < powf(Points_at_level, DIMENSIONS)){

		// This way we can obtain the global index of the cube vertex from the cube vertex's relative position
		uint32_t main_node_idx = 0;

		  
		for (uint32_t j = 0; j < DIMENSIONS; j++){
			main_node_idx += floor(positive_rem(globalID, pow(Points_at_level, j + 1)) / powf(Points_at_level, j)) * powf(PtsPerDim, j) * rescaling;
		}

		// find the dyadic cube indeces and do the wavelet transform at the same time
		 
		for(uint32_t d = 0; d < DIMENSIONS; d++){
			for(uint32_t k = 0; k < powf(2,DIMENSIONS);k++){
				if( floor(positive_rem(k, pow(2, d + 1)) / pow(2, d))  == 0 ){ // this part decides whether this is an approximation node
					uint32_t approx_coef_idx = main_node_idx;
					uint32_t detail_coef_idx;

					for (uint32_t j = 0; j < DIMENSIONS; j++){
						approx_coef_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(PtsPerDim, j) * rescaling / 2;
					}

					detail_coef_idx = approx_coef_idx + pow(PtsPerDim, d) * rescaling / 2;

					_1D_WVLET<T>(PDF[approx_coef_idx], PDF[detail_coef_idx]);
				}
			}
		}

		// Now, we will recheck all the wavelet coefficients and compare with the threshold
		Activate_node[main_node_idx].node = main_node_idx;

		for(uint32_t k = 1; k < powf(2,DIMENSIONS); k++){

			uint32_t temp_idx = main_node_idx;
			
			for (uint32_t j = 0; j < DIMENSIONS; j++){
				temp_idx += floor(positive_rem(k, pow(2, j + 1)) / pow(2, j)) * pow(PtsPerDim, j) * rescaling / 2;
			}

			Activate_node[temp_idx].node = temp_idx;

			if(abs(PDF[temp_idx]) >= TOLERANCE_AMR){
				Activate_node[temp_idx].AMR_selected = 1;
			}
		}
	}
}


int16_t ADAPT_MESH_REFINEMENT_nD(const thrust::host_vector<TYPE>& H_PDF, thrust::device_vector<TYPE>* D__PDF, std::vector<TYPE>* AdaptPDF, const gridPoint* H_Mesh, std::vector<gridPoint>* AdaptGrid, const int LvlFine, const int LvlCoarse, const int PtsPerDim) {
	// Final AMR procedure

	uint32_t Total_Points = pow(PtsPerDim, DIMENSIONS);

	uint32_t rescaling = 2;

	thrust::host_vector<AMR_node_select> 	H__Node_selection(Total_Points, {0,0});
	thrust::device_vector<AMR_node_select>	D__Node_selection = H__Node_selection;

	for (uint32_t k = 0; k < LvlFine - LvlCoarse; k++) {

		uint32_t Points_at_level  = PtsPerDim / rescaling;

		uint32_t Threads = fmin(THREADS_P_BLK, pow(Points_at_level, DIMENSIONS));
		uint32_t Blocks  = floor((pow(Points_at_level, DIMENSIONS) - 1) / Threads) + 1;

		D__Wavelet_Transform__F<TYPE> <<<Blocks, Threads>>> (rpc((*D__PDF),0), rpc(D__Node_selection,0),PtsPerDim,Points_at_level,rescaling);
		gpuError_Check(cudaDeviceSynchronize());
		
		rescaling *= 2;
	}

	thrust::sort(thrust::device, D__Node_selection.begin(),D__Node_selection.end());
	H__Node_selection = D__Node_selection;

	uint32_t counter = 0;
	while(H__Node_selection[counter].AMR_selected == 1){
		counter++;
	}

	AdaptGrid->resize(counter);
	AdaptPDF ->resize(counter);

	#pragma omp parallel for
	for(int32_t k = 0; k < counter; k++){
		AdaptGrid->at(k) = H_Mesh[H__Node_selection[k].node];
		AdaptPDF ->at(k) = H_PDF [H__Node_selection[k].node];
	}

	return 0;
}
#endif