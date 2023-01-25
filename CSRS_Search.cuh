#ifndef __CSRS_SEARCH_CUH__
#define __CSRS_SEARCH_CUH__

#include "Classes.cuh"

#include <ctime>						// clock
#include <cmath>						// math routines
#include <cstring>						// C string ops
#include <fstream>						// file I/O

#include "SRStree.h"
#include "SRStree_CPU.h"
#include "SRStree_GPU.h"

using namespace std;

/// <summary>
/// This function computes the "pos" neighbors inside a given radius for each "query" point. It makes use of the CSRS algorithm and it's dimension-independent.
/// </summary>
/// <param name="pos">- Particles where we search  </param>
/// <param name="ref_nb">- Number of these </param>
/// <param name="query">- Query points (a.k.a., the grid points)</param>
/// <param name="query_nb">- Number of grid points</param>
/// <param name="query_radius">- Search radius </param>
/// <param name="maxNeighborNum">- maximum number of neighbors</param>
/// <param name="ret_indexes">- array of indeces for the neighbors</param>
/// <param name="ret_dists">- Distances...</param>
/// <param name="ret_neigh">- Number of neighbors nearby </param>
void _CSRS_GeneralSearch(float*						 pos, 
						 int						 ref_nb, 
						 float*						 query, 
						 int						 query_nb, 
						 const float				 query_radius, 
						 int						 &maxNeighborNum,
						 thrust::host_vector<int>*	 out_ret_indexes,
						 thrust::host_vector<float>* out_ret_dists,
						 thrust::host_vector<int>*	 out_ret_neigh)
{
	int			bucket_size = 25;
	int			iterations	= 1;				// iterations of random movements
	const int	dim			= DIMENSIONS;

	//--------------------------------------------------------------------------------------
	//	Calculate bbx and set query radius - r													(THIS IS KNOWN WHEN WE DEFINE THE GRID!!)
	float* bnds_lo = new float[dim];
	float* bnds_hi = new float[dim];

	for (int d = 0; d < dim; d++) {			// find smallest enclosing rectangle for the data points
		float lo_bnd = pos[dim * 0 + d];		// lower bound on dimension d
		float hi_bnd = pos[dim * 0 + d];;		// upper bound on dimension d
		for (int j = 0; j < ref_nb; j++) {
			if (pos[dim * j + d] < lo_bnd) lo_bnd = pos[dim * j + d];		// note modification to make sure we stay inside the grid
			else if (pos[dim * j + d] > hi_bnd) hi_bnd = pos[dim * j + d];	// note modification to make sure we stay inside the grid
		}
		bnds_lo[d] = lo_bnd;	bnds_hi[d] = hi_bnd;
	}
	float sum = 0.0;
	for (int d = 0; d < dim; d++) {
		sum += (bnds_hi[d] - bnds_lo[d]) * (bnds_hi[d] - bnds_lo[d]);
	}
	float bbxDiagonalLength = sqrt(sum);

	//--------------------------------------------------------------------------------------
	//	Create tree structure in CPU and transfer it to GPU	
	SRStree*		the_tree	 = new SRStree(pos, ref_nb, dim, bucket_size);

	SRStree_CPU*	the_tree_cpu = new SRStree_CPU;
					the_tree_cpu ->Reconstruct(the_tree);	//reorganize nodes of the tree

	SRStree_GPU*	the_tree_gpu = new SRStree_GPU;
					the_tree_gpu ->Construct(the_tree_cpu, the_tree_cpu->GetNumNodes(), pos, ref_nb);	// transfer tree onto GPU
	//--------------------------------------------------------------------------------------
	//	Well align the tree nodes
	the_tree_gpu->SortNodes();
	//--------------------------------------------------------------------------------------
	//	Tight fit the bounding boxes
	the_tree_gpu->RefitAABB();

	//--------------------------------------------------------------------------------------
	//	Allocate memory for query points and output results 
	//int maxNeighborNum = 150;	//	for specifying the maximally returned number of neighbors in SRS
								//	When the resultant number of neighbors is greater than this number,
								//	our program will randomly pick this number of neighbors to report as results.
	//--------------------------------------------------------------------------------------
	float* gpu_queries;			//	for query points
	cudaMalloc((void**)&gpu_queries, sizeof(float) * query_nb * dim);
	cudaMemcpy(gpu_queries, query, sizeof(float) * query_nb * dim, cudaMemcpyHostToDevice);
	//--------------------------------------------------------------------------------------
	int* gpu_ret_indexes;	//	for returning indices of resultant points -- index starting from zero
	float* gpu_ret_dist;    //	for returning the squared distances of every neighbor
	int* gpu_neigh;			//	for returning the numbers of neighbors for each query point
	cudaMalloc((void**)&gpu_ret_indexes, sizeof(int) * query_nb * maxNeighborNum);
	cudaMalloc((void**)&gpu_ret_dist, sizeof(float) * query_nb * maxNeighborNum);
	cudaMalloc((void**)&gpu_neigh, sizeof(int) * query_nb);
	cudaMemset(gpu_ret_indexes, 0xffffffff, query_nb * maxNeighborNum * sizeof(int));
	cudaMemset(gpu_neigh, 0xffffffff, query_nb * sizeof(int));

	//--------------------------------------------------------------------------------------
	//	Copy query points onto GPU
	//	cudaMemcpy(gpu_queries, query, sizeof(float) * query_nb * dim, cudaMemcpyHostToDevice);

	//long time = clock();
	//printf("Computation is started ... ...\n");
	//--------------------------------------------------------------------------------------
	//	Randomly move points and conduct the queries of SRS
	bool bRes;
	for (int i = 0; i < iterations; i++) {

		//printf("Iteration: %d \n", i);
		bRes = the_tree_gpu->Search(gpu_queries, query_nb, query_radius, maxNeighborNum, gpu_ret_indexes, gpu_ret_dist, gpu_neigh);
		if (!bRes) { printf("Running out of memory!\n"); break; }
	}
	//printf("Computation is completed (in %ld ms)!\n", clock() - time);

	//--------------------------------------------------------------------------------------
	//	Output results
	int* ret_indexes = (int*)malloc(sizeof(int) * query_nb * maxNeighborNum);
	float* ret_dists = (float*)malloc(sizeof(float) * query_nb * maxNeighborNum);
	int* ret_neigh = (int*)malloc(sizeof(int) * query_nb);
	cudaMemcpy(ret_indexes, gpu_ret_indexes, sizeof(int) * query_nb * maxNeighborNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(ret_dists, gpu_ret_dist, sizeof(float) * query_nb * maxNeighborNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(ret_neigh, gpu_neigh, sizeof(int) * query_nb, cudaMemcpyDeviceToHost);
	//----------------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------
	out_ret_indexes->insert(out_ret_indexes->end(), &ret_indexes[0], &ret_indexes[query_nb * maxNeighborNum]);
	out_ret_dists->insert(out_ret_dists->end(), &ret_dists[0], &ret_dists[query_nb * maxNeighborNum]);
	out_ret_neigh->insert(out_ret_neigh->end(), &ret_neigh[0], &ret_neigh[query_nb]);

	//--------------------------------------------------------------------------------------
	free(ret_indexes);	free(ret_dists);	free(ret_neigh);
	printf("Finished.\n\n\n");

	//--------------------------------------------------------------------------------------
	//	Free GPU memory
	cudaFree(gpu_queries);
	cudaFree(gpu_ret_indexes);
	cudaFree(gpu_ret_dist);
	cudaFree(gpu_neigh);
	//--------------------------------------------------------------------------------------
	//	Free memeory of trees
	delete the_tree;    delete the_tree_cpu;	delete the_tree_gpu;
}

#endif