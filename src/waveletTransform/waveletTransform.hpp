#ifndef __WAVELET_TRANSFORM_HPP__
#define __WAVELET_TRANSFORM_HPP__

#include "include/headers.hpp"
#include "mesh/Domain.hpp"
#include "mesh/Particle.hpp"
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
	const uint64_t in_rescaling;
	const uintType in_nodes_per_dim;
	const uint64_t total_signal_nodes;
	const uint16_t dimensions;
	const cartesianMesh boundingBox;
public:
	deviceFunction void operator()(const uint64_t global_id) const;
};


class get_nodes_above_threshold {
public:
	floatType* signal;
	uint64_t* assigned_node_indeces;
	uintType* assigned_node_markers;
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
	hostFunction 
		waveletTransform();

	hostFunction 
		~waveletTransform();

private:
	// Input signal for wavelet transform: 
	std::unique_ptr<floatType[]> m_initialSignal;
	deviceUniquePtr<floatType> m_initialSignal_dvc;
	
	// Parameters
	uint16_t m_minRefinementLevel;
	uint16_t m_maxRefinementLevel;

	// Output signals
	std::unique_ptr<floatType[]> m_transformedSignal;
	std::unique_ptr<floatType[]> m_thresholdCutoffTransformedSignal;

	deviceUniquePtr<floatType> m_transformedSignal_dvc;
	deviceUniquePtr<floatType> m_thresholdCutoffTransformedSignal_dvc;

	// Output AMR arrays for indeces and assigned nodes. These are not externally-assigned, just here
	std::unique_ptr<uint64_t[]> m_assignedNodeIndeces;
	std::unique_ptr<uintType[]> m_assignedNodeMarkers;

	deviceUniquePtr<uint64_t> m_assignedNodeIndeces_dvc;
	deviceUniquePtr<uintType> m_assignedNodeMarkers_dvc;

	// other signal definitions:
	uint16_t __signal_dimension;
	cartesianMesh m_signalDomain;

public:
	hostFunction
		void set_signal_dimension(uint16_t input);

	hostFunction deviceFunction
		uint16_t signal_dimension() const;

	hostFunction
		void setMinRefinementLevel(uint16_t input);

	hostFunction deviceFunction
		uint16_t min_refinement_level() const;

	hostFunction
		void setMaxRefinementLevel(uint16_t input);

	hostFunction deviceFunction
		uint16_t max_refinement_level() const;

	hostFunction
		uint16_t set_initial_signal_host2dvc(const floatType* input_signal);

	hostFunction
		uint16_t setInitialSignal_dvc2dvc(const floatType* inputSignal_dvc);

	hostFunction deviceFunction
		uintType nodes_per_dim() const;

	hostFunction deviceFunction
		uint64_t total_signal_nodes() const;

	hostFunction
		floatType* initialSignal_ptr() const;

	hostFunction
		floatType* initialSignal_dvcPtr() const;

	hostFunction
		floatType* transformed_signal() const;

	hostFunction
		floatType* transformed_signal_dvc() const;

	hostFunction
		uint64_t* assigned_node_indeces() const;

	hostFunction
		uintType* assigned_node_markers() const;

	hostFunction
		uint64_t* assigned_node_indeces_dvc() const;

	hostFunction
		uintType* assigned_node_markers_dvc() const;

	hostFunction
		void computeWaveletTransform();

	hostFunction
		uintType sorted_assigned_nodes();

	hostFunction
		void setSignalDomain(const cartesianMesh& signalDomain);
};

#endif