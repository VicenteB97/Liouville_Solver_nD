#ifndef __ADAPT_MESH_CUH__
#define __ADAPT_MESH_CUH__

#include "Classes.cuh"

// We'll optimize the AMR for 1,2,3 dimensions and then leave the complex part for larger dimensions

class MeshId {
public:
	int indeces[DIMENSIONS];
};

// ---------------------- FUNCTIONS FOR GRADIENT - BASED AMR ---------------------- //
inline unsigned int positive_rem(const int a, const int b){
	return (a % b + b) % b;
} 

inline void _1D_WVLET(double& s1, double& s2){
	double aux;
	aux = 0.5*(s1+s2);
	s2 	= s1-s2;
	s1 	= aux;
}

// This function computes the Haar tensor wavelet in general dimensions 
void ND_WAVELET(std::vector<double>& cube){ // THIS WORKS !!!

	std::vector<unsigned int> base_indx ((unsigned int) powf(2, DIMENSIONS - 1));
	// get a vertex of the cube and write it in "binary"

	for (unsigned int d = 0; d < DIMENSIONS; d++){

		unsigned int num_vrtx = 0;

		// Select the base indeces for the dyadic cube
		for (unsigned int k = 0; k < pow(2, DIMENSIONS); k++){
			if( floor(positive_rem(k, pow(2, d + 1)) / pow(2, d))  == 0 ){
				base_indx[num_vrtx] = k;
				num_vrtx++;
			}
		}

		for (unsigned int k = 0; k < pow(2, DIMENSIONS - 1); k++){
			unsigned int idx1 = base_indx[k],
						 idx2 = base_indx[k] + pow(2, d);

			_1D_WVLET(cube[idx1], cube[idx2]);
		}
	}
	base_indx.clear();

}

std::vector<int> _nD_MultiLvlWavelet(const thrust::host_vector<double> PDF, const int LvlFine, const int LvlCoarse, const double tol, const int PtsPerDim) { // Extending the simple Haar wavelet transform to a multi-level transform

	std::vector<int> 	out;										// output
	std::vector<double> aux_signal 		(pow(2, DIMENSIONS));		// auxiliary vector to hold our dyadic cube entries
	std::vector<int> 	cube_indeces 	(pow(2, DIMENSIONS));		// auxiliary vector to hold our dyadic cube indeces

	unsigned int 		Total_Points = pow(PtsPerDim, DIMENSIONS);  // points per dimension at level...

	for (int k = 0; k < LvlFine - LvlCoarse - 1; k++) {

		int rescaling 		= (int)pow(2, k + 1);
		int Points_at_level = (int)pow(2, (LvlFine - (k + 1)));

		// What cube am I at?
		for (unsigned int i = 0; i < pow (Points_at_level, DIMENSIONS); i++){

			// This way we can obtain the global index of the cube vertex from the cube vertex position
			int ii = 0; 
			for (unsigned int j = 0; j < DIMENSIONS; j++){
				ii += floor(positive_rem(i, powf(Points_at_level, j + 1)) / powf(Points_at_level, j)) * rescaling;
			}

			// Create your dyadic cube
				// main cube vertex
				cube_indeces[0] = ii;
				aux_signal[0] 	= PDF[ii];

				// global indeces for each vertex of the cube
				for (unsigned int l = 1; l < pow(2, DIMENSIONS); l++){
					cube_indeces[l] = ii;

					for (unsigned int j = 0; j < DIMENSIONS; j++){
						cube_indeces[l] += floor(positive_rem(l, powf(2, j + 1)) / powf(2, j)) * pow(PtsPerDim, j) * rescaling / 2;
					}					

					aux_signal[l] = PDF[cube_indeces[l]];
				}

			// Now we call the gradient-based AMR for this cube
			ND_WAVELET(aux_signal);

			// Now we analyze the error values in order to choose what indeces we keep
			for (unsigned int l = 1; l < pow(2, DIMENSIONS); l++){
				if (abs(aux_signal[l]) > tol){
					out.push_back(cube_indeces[l]);
				}
			}
		}
	}
	return out;
}

void ADAPT_MESH_REFINEMENT_nD(const thrust::host_vector<double>& H_PDF, std::vector<double>* AdaptPDF, const gridPoint* H_Mesh, std::vector<gridPoint>* AdaptGrid, const int LvlFine, const int LvlCoarse, const int PtsPerDim) {
	// Final AMR procedure

	std::vector<int> Grid = _nD_MultiLvlWavelet(H_PDF, LvlFine, LvlCoarse, pow(10, -5), PtsPerDim);
	int g_length = Grid.size();

	for (int i = 0; i < g_length; i++) {
		AdaptGrid->push_back(H_Mesh[Grid[i]]);
		AdaptPDF ->push_back(H_PDF[Grid[i]]);
	}
}

// ----------------- THE FOLLOWING FUNCTIONS PERFORM THE WAVELET-BASED AMR ------------------//

/// <summary>
/// 
/// </summary>
/// <param name="PDF"></param>
/// <param name="LvlFine"></param>
/// <param name="LvlCoarse"></param>
/// <param name="tol"></param>
/// <param name="PtsPerDim"></param>
/// <returns></returns>
std::vector<MeshId> MultiLvlWavelet(const thrust::host_vector<double> PDF, const int LvlFine, const int LvlCoarse, const double tol, const int PtsPerDim) { // Extending the simple Haar wavelet transform to a multi-level transform

	std::vector<MeshId> out;
	std::vector<double> aux_signal(pow(2, DIMENSIONS));

	for (int k = 1; k < LvlFine - LvlCoarse; k++) {
		int rescaling = pow(2, k - 1);

		for (int j = 0; j < PtsPerDim / (2 * rescaling); j++) {

			int jj = 2 * j * rescaling;

			for (int i = 0; i < PtsPerDim / (2 * rescaling); i++) {

				int ii = 2 * i * rescaling;

				aux_signal[0] = PDF[ii + PtsPerDim * jj];
				aux_signal[1] = PDF[ii + rescaling + PtsPerDim * jj];
				aux_signal[2] = PDF[ii + PtsPerDim * (jj + rescaling)];
				aux_signal[3] = PDF[ii + rescaling + PtsPerDim * (jj + rescaling)];

				ND_WAVELET(aux_signal);

				if (k == LvlFine - LvlCoarse - 1) {
					out.push_back({ ii, jj });
				}
					
				// generalize this!!!
				if (abs(aux_signal[1]) > tol) {
					out.push_back({ ii + rescaling , jj });
				}
				else if (abs(aux_signal[2]) > tol) {
					out.push_back({ ii , jj + rescaling });
				}
				else if (abs(aux_signal[3]) > tol) {
					out.push_back({ ii + rescaling , jj + rescaling });
				}
			}
		}
	}
	return out;
}

/// <summary>
/// @brief This function returns the grid nodes that are "relevant enough" according to the Haar wavelet transform.
/// </summary>
/// <param name="H_PDF"> - PDF values at the grid </param>
/// <param name="AdaptPDF"> - PDF values at the relevant mesh points</param>
/// <param name="H_Mesh"> - Fixed, high-res. grid </param>
/// <param name="AdaptGrid"> - Returned, relevant grid nodes </param>
/// <param name="Grid"> - Not really valuable, just to fill the other arrays </param>
/// <param name="LvlFine"> - Finest level in the wavelet transform </param>
/// <param name="LvlCoarse"> - Coarsest level </param>
/// <param name="PtsPerDim"> - Poitns per dimension (2^LvlFine)</param>
void ADAPT_MESH_REFINEMENT(const thrust::host_vector<double>& H_PDF, std::vector<double>* AdaptPDF, const gridPoint* H_Mesh, std::vector<gridPoint>* AdaptGrid, const int LvlFine, const int LvlCoarse, const int PtsPerDim) {
	// Final AMR procedure

	std::vector<MeshId> Grid = MultiLvlWavelet(H_PDF, LvlFine, LvlCoarse, pow(10, -5), PtsPerDim);
	int g_length = Grid.size();

	for (int i = 0; i < g_length; i++) {

		// Re-write this for the general-dimensional case

		int ii = Grid[i].indeces[0];
		int jj = Grid[i].indeces[1];

		AdaptGrid->push_back(H_Mesh[ii + PtsPerDim * jj]);
		AdaptPDF ->push_back(H_PDF[ ii + PtsPerDim * jj]);
	}
}

#endif