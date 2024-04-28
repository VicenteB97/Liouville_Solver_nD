#pragma once

// C++/C libraries used
#include <fstream>      // Files output
#include <iostream>     // Command Line input/output
#include <vector>       // vector class is widely used
#include <cmath>        // C library (used for mathematical stuff)
#include <numeric>      // OPTIONAL?
#include <algorithm>    // OPTIONAL?
#include <cstdint>      // C library. Allows the use of specific int and unsigned int types, such as uint16_t
#include <string>       // For strings
#include <omp.h>        // For simple use of thread-parallel computations
#include <chrono>       // Timing the execution of functions
#include <limits>		// To get limits from variables
#include <thread>		// For multithreading
#include <atomic>		// For multithreading atomic functions

// Headers for the CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Thrust libraries
#include <thrust/host_vector.h>         // To use the host vector class. Equivalent to std::vector
#include <thrust/device_vector.h>       // GPU-equivalent of std::vector
#include <thrust/reduce.h>              // To compute the parallel reduction of vectors
#include <thrust/execution_policy.h>    // Important to specify where the reduction is taking place
#include <thrust/functional.h>          // OPTIONAL?
#include <thrust/transform.h>           // To multiply vectors using a parallel reduction-like fashion
#include <thrust/fill.h>                // Explicitly fill host/device functions with a value
#include <thrust/sort.h>                // GPU-based sorting
#include <thrust/extrema.h>             // Compute max. and min. of GPU arrays
#include <thrust/inner_product.h>		// To compute the inner product of GPU arrays

#include "Case_definition.cuh"
#include "utils/others.cuh"
#include "utils/numeric_defs.cuh"
#include "utils/error_functs.cuh"
#include "utils/error_msgs.cuh"