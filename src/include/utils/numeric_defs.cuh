#pragma once

// Maximum simulation file size
#define MAX_FILE_SIZE_B (uint64_t) 4*1024*1024*1024

// Check the dimensionality of the simulations to use "optimal" data types
#if PHASE_SPACE_DIM < 4
	#define intType int32_t
	#define uintType uint32_t
	#define ERASE_dPDF false
	#define ERASE_auxVectors false
#else
	#define intType int64_t
	#define uintType uint64_t
	#define ERASE_dPDF true
	#define ERASE_auxVectors false
#endif

// Define the number of threads per block (128 for consumer GPUs such as the RTX3060 or Quadro RTX4000)
#define THREADS_P_BLK 128

// In low-impact CUDA kernels we process more than one element at a time 
#define ELEMENTS_AT_A_TIME 4

// Mathematical definitions we need
#define M_PI 3.14159265358979323846
#define ptSEARCH_THRESHOLD 10000000 // The min. number of particles per sample where we use the counting sort search (not worth it yet)