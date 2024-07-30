#ifndef __WAVELET_TRANSFORM_HPP__
#define __WAVELET_TRANSFORM_HPP__

#include "include/headers.hpp"
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief (DEVICE FUNCTION) Compute a 1D Haar wavelet transform
/// @param s1 
/// @param s2 
/// @return
deviceFunction inline void haar_wavelet(floatType& s1, floatType& s2);

class single_block_single_level_wavelet_transform {
public:
	floatType* signal;
	uint64_t in_rescaling;
	uint32_t in_nodes_per_dim;
	uint64_t total_signal_nodes;
	uint16_t dimensions;
public:
	deviceFunction void operator()(const uint64_t global_id) const;
};


class get_nodes_above_threshold {
public:
	floatType* signal;
	uint64_t* assigned_node_indeces;
	uint32_t* assigned_node_markers;
	const uint64_t rescaling;
	const uint64_t nodes_per_dim;
	const uint64_t total_signal_nodes;
	const double tolerance;
	const uint16_t dimensions;
public:
	deviceFunction void operator()(const uint64_t global_id) const;
};


class waveletTransform {
public:
	// Default constructor and destrucor
	waveletTransform();
	~waveletTransform();

private:
	// Input signal for wavelet transform: 
	floatType* __initial_signal;
	floatType* __initial_signal_dvc;

	// Parameters
	uint16_t __min_refinement_level;
	uint16_t __max_refinement_level;

	// Output signals
	floatType* __transformed_signal;
	floatType* __transformed_signal_dvc;
	floatType* __threshold_cutoff_transformed_signal;
	floatType* __threshold_cutoff_transformed_signal_dvc;

	// Output AMR arrays for indeces and assigned nodes. These are not externally-assigned, just here
	uint64_t* __assigned_node_indeces;
	uint64_t* __assigned_node_indeces_dvc;
	uint32_t* __assigned_node_markers;
	uint32_t* __assigned_node_markers_dvc;

	// other signal definitions:
	uint16_t __signal_dimension;

public:
	hostFunction
		void set_signal_dimension(uint16_t input);

	hostFunction deviceFunction
		uint16_t signal_dimension() const;

	hostFunction
		void set_min_refinement_level(uint16_t input);

	hostFunction deviceFunction
		uint16_t min_refinement_level() const;

	hostFunction
		void set_max_refinement_level(uint16_t input);

	hostFunction deviceFunction
		uint16_t max_refinement_level() const;

	hostFunction
		void set_initial_signal_host2dvc(const floatType* input_signal);

	hostFunction
		void set_initial_signal_dvc2dvc(const floatType* input_signal_dvc);

	hostFunction deviceFunction
		uint32_t nodes_per_dim() const;

	hostFunction deviceFunction
		uint64_t total_signal_nodes() const;

	hostFunction
		floatType* initial_signal() const;

	hostFunction
		floatType* initial_signal_dvc() const;

	hostFunction
		floatType* transformed_signal() const;

	hostFunction
		floatType* transformed_signal_dvc() const;

	hostFunction
		uint64_t* assigned_node_indeces() const;

	hostFunction
		uint32_t* assigned_node_markers() const;

	hostFunction
		uint64_t* assigned_node_indeces_dvc() const;

	hostFunction
		uint32_t* assigned_node_markers_dvc() const;

	hostFunction
		void compute_wavelet_transform();
};

#endif