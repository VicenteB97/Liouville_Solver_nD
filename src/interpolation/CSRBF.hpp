#ifndef __CSRBF_HPP__
#define __CSRBF_HPP__

#include "include/headers.hpp"

deviceFunction inline floatType RBF(const floatType& SuppRBF, const floatType& inputNormalized) {

#if PHASE_SPACE_DIMENSIONS == 1
	const double Mass_RBF = 0.333383333333333;

#elif PHASE_SPACE_DIMENSIONS == 2
	const double Mass_RBF = 0.4487989332761852; // this is actually the: int_0^1 phi(r)r dr

#elif PHASE_SPACE_DIMENSIONS == 3
	const double Mass_RBF = 0.2991993007905369;

#elif PHASE_SPACE_DIMENSIONS == 4
	const double Mass_RBF = 0.19582548282938347;
#endif

	return (floatType)powf(fmaxf(0, 1 - inputNormalized), 4) * (4 * inputNormalized + 1) / Mass_RBF / powf(SuppRBF, PHASE_SPACE_DIMENSIONS); // We multiply by this last factor to get the L1-normalized RBF

}
#endif