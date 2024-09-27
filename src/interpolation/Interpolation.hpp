//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//														//
//		Written by: Vicente José Bevia Escrig			//
//		Mathematics Ph.D. student (2020-2024) at:		//
//		Instituto de Matemática Multidisciplinar,		//
//		Universitat Politècnica de València, Spain		//
//														//
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
#ifndef __INTERPOLATION_HPP__
#define __INTERPOLATION_HPP__

#include "CSRBF.hpp"
#include "Remeshing.hpp"
#include "include/headers.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ConjugateGradientEngine {
private:
	deviceUniquePtr<floatType> m_R_dvc;
	deviceUniquePtr<floatType> m_temp_dvc;
	deviceUniquePtr<floatType> m_AP_dvc;
	deviceUniquePtr<floatType> m_P_dvc;

public:
	// Constructor
	ConjugateGradientEngine(uint32_t sizeCount = 0) :
			m_R_dvc(deviceUniquePtr<floatType>(sizeCount)), 
			m_temp_dvc(deviceUniquePtr<floatType>(sizeCount)),
			m_AP_dvc(deviceUniquePtr<floatType>(sizeCount)),
			m_P_dvc(deviceUniquePtr<floatType>(sizeCount)) {};
	// Methods
	uint16_t execute(
		deviceUniquePtr<floatType>& inputOutputVector,
		const deviceUniquePtr<floatType>& targetVector,
		const deviceUniquePtr<int64_t>& matrixIndeces,
		const deviceUniquePtr<floatType>& matrixValues,
		const uint16_t maxNeighbors,
		const uint16_t maxIterations = __DEFAULT_CONJUGATE_GRADIENT_MAX_ITERS__,
		const double interpolationTolerance = __CGrad_TOLERANCE__
	); // Interpolate
};

class COOMatVecMultiplication_dvc {
public:
	floatType* x_dvc;
	const floatType* x0_dvc;
	const int64_t* matrixIdxs_dvc;
	const floatType* matrixEntries_dvc;
	const uint64_t totalLength;
	const intType maxNeighbors;

public:
	deviceFunction
	void operator()(const uint64_t global_id);
};

class vectorUpdate_dvc {
public:
	floatType* x_dvc;
	const floatType* x0_dvc;
	const floatType scalar;
	const floatType* directionVector;
	const uint64_t totalLength;

public:
	deviceFunction
	void operator()(const uint64_t global_id);
};

template<typename _Ty>
_Ty innerProduct_dvc(const _Ty* vectorOne, const _Ty* vectorTwo, const _Ty initialValue) {
	#ifdef USECUDA
	thrust::device_ptr<_Ty> vectorOne_thrustPtr(vectorOne);
	thrust::device_ptr<_Ty> vectorTwo_thrustPtr(vectorTwo);

	return thrust::inner_product(
		thrust::device, 
		vectorOne_thrustPtr.begin(),
		vectorOne_thrustPtr.end(),
		vectorTwo_thrustPtr.begin(),
		initialValue);
	#endif // USECUDA
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif