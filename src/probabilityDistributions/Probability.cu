#include "Probability.cuh"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Distributions::Distributions() {
	params[0] = 0;
	params[1] = 0;

	Name = 'D';

	isTruncated = true;

	trunc_interval[0] = 0;
	trunc_interval[1] = 0;

	num_Samples = 1;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief This function creates the IC PDF. We assume it is the tensor product of Normal Distributions
/// @param Points_per_dimension 
/// @param cartesianMesh 
/// @param PDF_value 
/// @param IC_dist_parameters 
int16_t PDF_INITIAL_CONDITION(const cartesianMesh& problem_domain, std::shared_ptr<floatType[]>& PDF_value, const Distributions* IC_dist_parameters) {

	std::vector<floatType> temp_val(problem_domain.nodes_per_dim() * PHASE_SPACE_DIMENSIONS);

	for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
		// Create the arrays for each dimension!
		floatType expectation = IC_dist_parameters[d].params[0], std_dev = IC_dist_parameters[d].params[1],
			x0 = fmax(problem_domain.boundary_inf().dim[d], expectation - 8 * std_dev),
			xF = fmin(problem_domain.boundary_sup().dim[d], expectation + 8 * std_dev),
			rescale_CDF = 1;

		if (IC_dist_parameters[d].Name == 'N' || IC_dist_parameters[d].Name == 'n') {
			auto dist = boost::math::normal_distribution(expectation, std_dev);

			if (IC_dist_parameters[d].isTruncated) {
				x0 = fmaxf(x0, IC_dist_parameters[d].trunc_interval[0]);
				xF = fminf(xF, IC_dist_parameters[d].trunc_interval[1]);

				rescale_CDF = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);
			}

			#pragma omp parallel for
			for (intType k = problem_domain.nodes_per_dim() * d; k < problem_domain.nodes_per_dim() * (d + 1); k++) {
				temp_val[k] = boost::math::pdf(dist, problem_domain.get_node((k - problem_domain.nodes_per_dim() * d) * pow(problem_domain.nodes_per_dim(), d)).dim[d]) / rescale_CDF;
			}
		}
	}

	std::cout << "Filling initial density...\n";

	#pragma omp parallel for
	for (intType k = 0; k < problem_domain.total_nodes(); k++) {
		floatType val = 1;
		for (uint16_t d = 0; d < PHASE_SPACE_DIMENSIONS; d++) {
			intType temp_idx = floor(positive_rem(k, pow(problem_domain.nodes_per_dim(), d + 1)) / pow(problem_domain.nodes_per_dim(), d));

			val *= temp_val[temp_idx + problem_domain.nodes_per_dim() * d];
		}
		PDF_value[k] = val;
	}

	std::cout << "Initial density successfully filled!\n";
	return EXIT_SUCCESS;
}


// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //
// ------------------ TO BUILD THE RANDOM PARAMETER MESH! ---------------------------- //
// ----------------------------------------------------------------------------------- //
// ----------------------------------------------------------------------------------- //

/// @brief 
/// @param random_var_sample_count 
/// @param PP 
/// @param Dist_params 
int16_t PARAMETER_VEC_BUILD(const int random_var_sample_count, Param_pair* PP, const Distributions Dist_params) {

	floatType expectation = Dist_params.params[0], std_dev = Dist_params.params[1];

	if (random_var_sample_count == 1) {
		*PP = { expectation, 1 };
		return 0;
	}

	floatType x0, xF, dx;

	// Normal case
	if (Dist_params.Name == 'N' || Dist_params.Name == 'n') {
		// Build the distribution:
		auto dist = boost::math::normal_distribution(expectation, std_dev);

		x0 = expectation - 6 * std_dev;
		xF = expectation + 6 * std_dev;


		if (Dist_params.isTruncated) {
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			floatType rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// cartesianMesh discretization
			dx = (xF - x0) / (random_var_sample_count - 1);

			for (int i = 0; i < random_var_sample_count; i++) {
				floatType x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (random_var_sample_count - 1);

		for (int i = 0; i < random_var_sample_count; i++) {
			floatType x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	// Uniform case
	if (Dist_params.Name == 'U' || Dist_params.Name == 'u') {

		x0 = expectation - sqrt(3) * std_dev;
		xF = expectation + sqrt(3) * std_dev;

		dx = (xF - x0) / (random_var_sample_count - 1);

		for (int i = 0; i < random_var_sample_count; i++) {
			floatType x = x0 + i * dx;
			PP[i] = { x , 1/(xF-x0)}; // other distributions could be used
		}

		return 0;
	}

	// Gamma case
	if (Dist_params.Name == 'G' || Dist_params.Name == 'g') {

		floatType shape = pow(expectation / std_dev, 2);
		floatType scale = pow(std_dev, 2) / expectation;

		auto dist = boost::math::gamma_distribution(shape, scale);

		x0 = fmaxf(0, expectation - 6 * std_dev);
		xF = expectation + 6 * std_dev;


		if (Dist_params.isTruncated) {
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			floatType rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// cartesianMesh discretization
			dx = (xF - x0) / (random_var_sample_count - 1);

			for (int i = 0; i < random_var_sample_count; i++) {
				floatType x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (random_var_sample_count - 1);

		for (int i = 0; i < random_var_sample_count; i++) {
			floatType x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	// Beta case
	if (Dist_params.Name == 'B' || Dist_params.Name == 'b') {

		floatType shape_1 = -expectation * (1 - (1 - expectation) / std_dev * expectation / std_dev);
		floatType shape_2 = (expectation - 1) * (1 - (1 - expectation) / std_dev * expectation / std_dev);

		auto dist = boost::math::beta_distribution(shape_1, shape_2);

		x0 = fmaxf(0, expectation - 6 * std_dev);
		xF = fminf(1, expectation + 6 * std_dev);


		if (Dist_params.isTruncated) {
			x0 = fmaxf(x0, Dist_params.trunc_interval[0]);
			xF = fminf(xF, Dist_params.trunc_interval[1]);

			// Re-scaling for the truncation of the random variables
			floatType rescale_cdf = boost::math::cdf(dist, xF) - boost::math::cdf(dist, x0);

			// cartesianMesh discretization
			dx = (xF - x0) / (random_var_sample_count - 1);

			for (int i = 0; i < random_var_sample_count; i++) {
				floatType x = x0 + i * dx;
				PP[i] = { x, boost::math::pdf(dist, x) / rescale_cdf }; // other distributions could be used
			}
			return 0;
		}

		dx = (xF - x0) / (random_var_sample_count - 1);

		for (int i = 0; i < random_var_sample_count; i++) {
			floatType x = x0 + i * dx;
			PP[i] = { x, boost::math::pdf(dist, x) }; // other distributions could be used
		}
		return 0;
	}

	std::cout << "Parameter distribution not recognized. Choose correctly please. \nExiting program...\n";
	return -1;
}
