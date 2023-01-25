#ifndef __CLASSES_CUH__
#define __CLASSES_CUH__

#include "Global_Declarations.cuh"


//----------------- CLASSES USED ---------------------------------
//----------------------------------------------------------------

// Grid points----------------------------------------------------
class gridPoint {
public:
	double position[DIMENSIONS];

	__host__ __device__ gridPoint operator+(const gridPoint& other) {

		gridPoint out;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			double aux = position[d];
			aux += other.position[d];
			out.position[d] = aux;
		}

		return out;

	}
	__host__ __device__ gridPoint operator-(const gridPoint& other) {
		gridPoint out;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			double aux = position[d];
			aux -= other.position[d];
			out.position[d] = aux;
		}

		return out;
	}
	__host__ __device__ bool operator==(const gridPoint& other) {
		bool out = true;

		for (unsigned int d = 0; d < DIMENSIONS; d++) {
			if (position[d] != other.position[d]) { out = false; }
		}

		return out;
	}
};

// Time + impulse: ----------------------------------------------
class Time_Impulse_vec {
public:
	double time;
	bool impulse;
	gridPoint impulse_vec;
	double impulse_std[DIMENSIONS];
	int samples[DIMENSIONS];

	bool operator < (const Time_Impulse_vec& other) const {
		return (time < other.time);
	}
};

// Parameters
class Param_vec {
public:
	double sample_vec[PARAM_DIMENSIONS];
	double Joint_PDF;
};

class Param_pair {
public:
	double sample, PDF;
};

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//------------------ FUNCTIONS TO BE USED ELSEWHERE IN THE CODE------------
//-------------------------------------------------------------------------

/// <summary>
/// Computes euclidean distance between two points in space
/// </summary>
/// <param name="P1"></param>
/// <param name="P2"></param>
/// <returns></returns>
__host__ __device__ double Distance(const gridPoint P1, const gridPoint P2) {

	double out = 0;

	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out += (P1.position[d] - P2.position[d]) * (P1.position[d] - P2.position[d]);
	}

	return sqrt(out);
}

/// <summary>
/// Computes the multiplication of a scalar with a point in space
/// </summary>
/// <param name="scalar"></param>
/// <param name="Point"></param>
/// <returns></returns>
__device__ gridPoint Mult_by_Scalar(double scalar, gridPoint Point) {
	gridPoint out;

	for (unsigned int d = 0; d < DIMENSIONS; d++) {
		out.position[d] = scalar * Point.position[d];
	}

	return out;
}


// THESE FUNCTIONS ARE ONLY USED IN THE CASE WHERE THE "NEW" POINT SEARCH ALGORITHM IS USED

/// <summary>
/// 
/// </summary>
/// <param name="Particles_solver"></param>
/// <param name="Particles_CSRS"></param>
/// <param name="length"></param>
void IDX_TRANSLATION_Query(const gridPoint* Particles_solver, float* Particles_CSRS, const int length) {
	for (int i = 0; i < length; i++) {
		for (int d = 0; d < DIMENSIONS; d++) {
			Particles_CSRS[i * DIMENSIONS + d] = Particles_solver[i].position[d];
		}
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="Particles_solver"></param>
/// <param name="Particles_CSRS"></param>
/// <param name="length"></param>
/// <param name="Real_Length"></param>
/// <param name="Limit_high"></param>
/// <param name="Limit_low"></param>
void IDX_TRANSLATION_Data(const gridPoint* Particles_solver, std::vector<float>* Particles_CSRS, const int length) {

	for (int i = 0; i < length; i++) {
		for (int d = 0; d < DIMENSIONS; d++) {
			Particles_CSRS->push_back(Particles_solver[i].position[d]);
		}
	}
}

#endif