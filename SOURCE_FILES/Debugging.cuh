#pragma once
#include "Classes.cuh"

// ONLY FOR DEBUGGING
//int* Index_array = new int[MaxNeighborNum * Total_Points];					// array of particle indeces
//double* Mat_entries = new double[MaxNeighborNum * Total_Points];				// distance matrix entries
//int* Num_Neighbors = new int[Total_Points];									// number of neighbors per particle
// double* Lambdas = new double[Total_Points];									// The lambdas obtained from solving the system



// Debug Lambdas from interpolation
//double* Lambdas = new double[Total_Particles];
//thrust::copy(GPU_lambdas.begin(), GPU_lambdas.end(), &Lambdas[0]);
//DEBUG_LAMBDAS_VEC(Lambdas, Adapt_Points, Random_Samples);
//delete[] Lambdas;

// Debug Interpolation Matrix
// double* Mat_entries = new double[Total_Points * MaxNeighborNum];
// thrust::copy(GPU_Mat_entries.begin(), GPU_Mat_entries.end(), &Mat_entries[0]);
// DEBUG_INTERPOLATION_MATRIX(Mat_entries, Total_Points, MaxNeighborNum);
// delete[] Mat_entries;

// Debug Indeces
/*int* Index_array = new int[Block_Particles * MaxNeighborNum];
thrust::copy(GPU_Index_array.begin(), GPU_Index_array.end(), &Index_array[0]);
DEBUG_INDEX_ARRAY(Index_array, Block_Particles, MaxNeighborNum);
delete[] Index_array;*/

template <class T>
void DEBUG_General(T Mat_entries, const int Total_Pts, const int Max_Neighbors) {
	std::ofstream myfile_0;
	myfile_0.open("CSRS_preps.csv");
	if (myfile_0.is_open()) {

		for (int k = 0; k < Total_Pts; k++) {
			for (int i = 0; i < Max_Neighbors; i++) {
				myfile_0 << Mat_entries[i + k * Max_Neighbors] << ",";
			}
			myfile_0 << "\n";
		}

		myfile_0.close();
		std::cout << "Completed!\n";
	}
	else {
		std::cout << "Failed!!\n";
	}
}


void DEBUG_INTERPOLATION_MATRIX(double* Mat_entries, const int Adapt_Points, const int MaxNeighborNum) {
	std::ofstream myfile_0;
	myfile_0.open("Matrix_info.csv");
	if (myfile_0.is_open()) {
		for (int k = 0; k < Adapt_Points; k++) {
			for (int i = 0; i < MaxNeighborNum - 1; i++) {
				myfile_0 << Mat_entries[i + k * MaxNeighborNum] << ",";
			}
			myfile_0 << Mat_entries[(k + 1) * MaxNeighborNum - 1] << "\n";
		}
		myfile_0.close();
		std::cout << "Completed!\n";
	}
	else {
		std::cout << "Failed!!\n";
	}
}
template<class T>
void DEBUG_LAMBDAS_VEC(std::vector<T>& Mat_entries, const int Adapt_Points, const int Random_Samples) {
	std::ofstream myfile_0;
	myfile_0.open("Lambdas_info.csv");
	if (myfile_0.is_open()) {
		for (int k = 0; k < Random_Samples; k++) {
			for (int i = 0; i < Adapt_Points - 1; i++) {
				myfile_0 << Mat_entries[i + k * Adapt_Points] << ",";
			}
			myfile_0 << Mat_entries[(k + 1) * Adapt_Points - 1] << "\n";
		}
		myfile_0.close();
		std::cout << "Completed!\n";
	}
	else {
		std::cout << "Failed!!\n";
	}
}

void DEBUG_INDEX_ARRAY(int* Mat_entries, const int Adapt_Points, const int MaxNeighborNum) {
	std::ofstream myfile_0;
	myfile_0.open("Idx_info.csv");
	if (myfile_0.is_open()) {
		for (int k = 0; k < Adapt_Points; k++) {
			for (int i = 0; i < MaxNeighborNum; i++) {
				myfile_0 << Mat_entries[i + k * MaxNeighborNum] << ",";
			}
			myfile_0 << std::endl;
		}
		myfile_0.close();
		std::cout << "Completed!\n";
	}
	else {
		std::cout << "Failed!!\n";
	}
}