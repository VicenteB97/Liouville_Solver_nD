#pragma once
#include "numeric_defs.cuh"

// This is for the thrust library
#define rpc(ans,offset) raw_pointer_cast(&ans[offset])

// This function is widely used throughout the code so...quite effective
/// @brief Host/Device function that computes the positive remainder (mod) between two integers
/// @param a This is the numerator
/// @param b This is the denominator
/// @return Returns mod(a,b)
__host__ __device__
inline uintType positive_rem(const uintType a, const uintType b) {
	return (a % b + b) % b;
}