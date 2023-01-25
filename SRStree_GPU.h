#ifndef __CUDA_SRSTREE_H__
#define __CUDA_SRSTREE_H__

#include "SRStree_CPU.h"

//----------------------------------------------------------------------
//	AABB-tree node in GPU
//
//		Nodes of the AABB-tree in GPU store the type, index, level, bounding box,
//		index of its parent, left child and right child, number of points contained
//		in bucket, point index to an point array, center and spread of the bounding
//      box.
//----------------------------------------------------------------------
struct CUDA_AABBNode
{
	int type;
	int id;
	int level;
	int parent, left, right;
	int n_pts;
	int indexes;
	float cd_bnds[DIMENSIONS * 2];
	float center[DIMENSIONS];
	float extent[DIMENSIONS];

	float volume;
};

class SRStree_GPU
{
public:
	~SRStree_GPU();

	void Construct(SRStree_CPU* tree, int num_nodes, float* data, int num_pts);
	bool Search(float* gpu_queries, int query_num, float radii, int MaxSearchNeigh, int* real_gpu_ret_indexes, float* real_gpu_ret_dist, int* real_gpu_neigh);
	bool SearchByDataPoints(float radii, int MaxSearchNeigh, int* real_gpu_ret_indexes, float* real_gpu_ret_dist, int* real_gpu_neigh);
	bool SearchNN(float* gpu_queries, int query_num, float radii, int* real_gpu_ret_index, float* real_gpu_ret_dist);
	void RefitAABB();
	void SortNodes();

	bool isRebuildNeeded();

	float* GetDataPoints() { return m_gpu_points; }
	void SetDataPoints(float* pos, int num);

private:
	CUDA_AABBNode* m_gpu_nodes;
	int* m_gpu_indexes;
	float* m_gpu_points;
	int* m_gpu_leafnodes;

	int* m_gpu_idx;
	int* m_gpu_level;
	int* m_gpu_start;
	int* m_gpu_end;

	int m_num_nodes;
	int m_num_points;
	int dimension;
	int m_num_leafnodes;
	int m_max_level;

	float volThrd;
	bool flag_rebuild;
};

#endif