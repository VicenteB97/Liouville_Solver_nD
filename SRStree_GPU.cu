#include "SRStree_GPU.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <float.h>
#include <cstdio>
#include <vector>
#include "math.h"
// #include <windows.h>
//#include <cutil_math.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/scan.h"
#include "thrust/unique.h"
#include "thrust/extrema.h"

using namespace std;


#define THREAD_PER_BLOCK 256
#define NUMBER_OF_BLOCK 32

#define BUCKET_SIZE 30

#define CUDA_STACK 16		// fixed size stack elements for each thread, increase as required. Used in BackTraversal.

//Round a / b to nearest higher integer value
int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeGridSize(int n, int blockSize, int& numBlocks, int& numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

SRStree_GPU::~SRStree_GPU()
{
	cudaFree(m_gpu_nodes);
	cudaFree(m_gpu_indexes);
	cudaFree(m_gpu_points);
	cudaFree(m_gpu_leafnodes);

	cudaFree(m_gpu_idx);
	cudaFree(m_gpu_level);
	cudaFree(m_gpu_start);
	cudaFree(m_gpu_end);
}



// Create the nodes again on the CPU. The nodes are laid out in the ordre of breath-first traversal (BFT).
// The nodes at the same level are stored together.
void SRStree_GPU::Construct(SRStree_CPU* tree, int num_nodes, float* data, int num_pts)
{
	m_num_points = num_pts;
	dimension = tree->GetDim();
	m_num_leafnodes = tree->m_num_leafnodes;
	m_max_level = tree->max_level;
	m_num_nodes = num_nodes;

	volThrd = 10.0;
	flag_rebuild = false;

	cudaMalloc((void**)&m_gpu_nodes, sizeof(CUDA_AABBNode) * num_nodes);
	cudaMemset(m_gpu_nodes, 0, sizeof(CUDA_AABBNode) * num_nodes);
	cudaMalloc((void**)&m_gpu_indexes, sizeof(int) * m_num_points);
	cudaMalloc((void**)&m_gpu_points, sizeof(float) * m_num_points * dimension);
	cudaMalloc((void**)&m_gpu_leafnodes, sizeof(int) * m_num_leafnodes);

	cudaMalloc((void**)&m_gpu_idx, sizeof(int) * num_nodes);
	cudaMalloc((void**)&m_gpu_level, sizeof(int) * num_nodes);
	cudaMalloc((void**)&m_gpu_start, sizeof(int) * (m_max_level + 1));
	cudaMalloc((void**)&m_gpu_end, sizeof(int) * (m_max_level + 1));


	vector <CUDA_AABBNode> cpu_nodes(num_nodes);
	vector <int> indexes(m_num_points);
	vector <AABBNode_CPU*> to_visit;
	vector<int> cpu_leafnodes;


	int cur_pos = 0;

	to_visit.push_back(tree->m_root);

	//Store the nodes in order of BFT
	while (to_visit.size()) {
		vector <AABBNode_CPU*> next_search;

		while (to_visit.size()) {
			AABBNode_CPU* cur = to_visit.back();
			to_visit.pop_back();

			int id = cur->id;

			cpu_nodes[id].id = cur->id;
			cpu_nodes[id].level = cur->level;
			cpu_nodes[id].parent = cur->_parent;
			if (cur->type == 0)
			{
				cpu_nodes[id].type = 0;
				cpu_nodes[id].left = cur->_left;
				cpu_nodes[id].right = cur->_right;
				for (int k = 0; k < 2 * dimension; k++)
				{
					cpu_nodes[id].cd_bnds[k] = cur->bnd_box[k];
				}
				for (int k = 0; k < dimension; k++)
				{
					cpu_nodes[id].center[k] = 0.5f * (cur->bnd_box[k] + cur->bnd_box[k + dimension]);
					cpu_nodes[id].extent[k] = 0.5f * (cur->bnd_box[k + dimension] - cur->bnd_box[k]);
				}

				float vol = 1.0;
				for (int k = 0; k < dimension; k++)
				{
					vol *= cur->bnd_box[k + dimension] - cur->bnd_box[k];
				}

				cpu_nodes[id].volume = vol;
			}

			if (cur->type == 1)
			{
				cpu_nodes[id].type = 1;
				cpu_nodes[id].n_pts = cur->bkt.size();

				cpu_leafnodes.push_back(id);

				for (int k = 0; k < 2 * dimension; k++)
				{
					cpu_nodes[id].cd_bnds[k] = cur->bnd_box[k];
				}
				for (int k = 0; k < dimension; k++)
				{
					cpu_nodes[id].center[k] = 0.5f * (cur->bnd_box[k] + cur->bnd_box[k + dimension]);
					cpu_nodes[id].extent[k] = 0.5f * (cur->bnd_box[k + dimension] - cur->bnd_box[k]);
				}

				float vol = 1.0;
				for (int k = 0; k < dimension; k++)
				{
					vol *= cur->bnd_box[k + dimension] - cur->bnd_box[k];
				}

				cpu_nodes[id].volume = vol;

				if (cur->bkt.size()) {
					for (unsigned int i = 0; i < cur->bkt.size(); i++)
					{
						indexes[cur_pos + i] = cur->bkt[i];
					}

					cpu_nodes[id].indexes = cur_pos;
					cur_pos += cur->bkt.size();
				}
				else {
					cpu_nodes[id].indexes = -1;
				}
			}

			if (cur->left)
				next_search.push_back(cur->left);

			if (cur->right)
				next_search.push_back(cur->right);
		}

		to_visit = next_search;
	}

	//Trasfer the nodes to GPU memory
	cudaMemcpy(m_gpu_nodes, &cpu_nodes[0], sizeof(CUDA_AABBNode) * num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_gpu_indexes, &indexes[0], sizeof(int) * indexes.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(m_gpu_points, data, sizeof(float) * m_num_points * dimension, cudaMemcpyHostToDevice);
	cudaMemcpy(m_gpu_leafnodes, &cpu_leafnodes[0], sizeof(int) * m_num_leafnodes, cudaMemcpyHostToDevice);
}

//Distance between two points
__device__ float PtToPt(float* pt1, float* pt2, int d)
{
	float dist = 0.0;

	for (int i = 0; i < d; i++)
	{
		dist += (pt2[i] - pt1[i]) * (pt2[i] - pt1[i]);
	}

	return dist;
}

//Distance from one point to an AABB
__device__ float PtToBbx(const float* bnds, float* pt, int d)
{
	float dist = 0.0;

	for (int i = 0; i < d; i++)
	{
		float temp = 0.0;
		if (pt[i] < bnds[i])
			temp = pt[i] - bnds[i];
		else if (pt[i] > bnds[i + d])
			temp = pt[i] - bnds[i + d];

		dist += temp * temp;
	}

	return dist;
}

//Distance between two AABBs
__device__ float BbxToBbx(const float* centerA, const float* extentA, const float* centerB, const float* extentB, int d)
{
	float dist = 0.0;

	for (unsigned int i = 0; i < d; i++)
	{
		float subCenter = abs(centerA[i] - centerB[i]);
		float addExtent = extentA[i] + extentB[i];
		float temp = subCenter - addExtent;
		if (temp > 0) {
			dist += temp * temp;
		}
	}

	return dist;
}

//Whether a point is inside an AABB or not
__device__ bool WithinBbx(const float* bnds, float* pt, int d)
{
	bool flag = true;
	for (int i = 0; i < d; i++)
	{
		bool f = false;
		if ((pt[i] >= bnds[i]) && (pt[i] <= bnds[i + d]))
			f = true;
		flag = flag && f;
	}

	if (flag)
	{
		return true;
	}
	else
		return false;
}

// Goes through all the nodes that are within "range"
__device__ void SearchAtNodeRange(int threadid, int index, unsigned int* idx, const CUDA_AABBNode* nodes, const int* indexes, int treedepth, float* pts, int dim, int cur, float radius, float range, int* ret_index, int* ret_indexvsleafid)
{
	// We'll use a fixed length stack, increase this as required
	__shared__ int to_visit[256][CUDA_STACK];
	int to_visit_pos = 0;

	to_visit[threadid][to_visit_pos++] = cur;

	//Traverse the tree in depth-first search so that the size of the stack 
	//is bounded by the tree's depth
	while (to_visit_pos) {
		//        int next_search[CUDA_STACK];
		//        int next_search_pos = 0;

		while (to_visit_pos) {
			cur = to_visit[threadid][to_visit_pos - 1];
			to_visit_pos--;

			if (nodes[cur].left == 0) {//cur is a leaf node
				for (int i = 0; i < nodes[cur].n_pts; i++) {
					int idxx = indexes[nodes[cur].indexes + i];

					float d = PtToBbx(&nodes[index].cd_bnds[0], &pts[dim * idxx], dim);

					if (d <= range)
					{
						unsigned int ind = atomicAdd(idx, 1);

						ret_index[ind] = idxx;
						ret_indexvsleafid[ind] = index;
					}
				}
			}
			else {//cur is a split node, then its children (left and right) needs to be checked
				float d = 0.0;
				d = BbxToBbx(&nodes[index].center[0], &nodes[index].extent[0],
					&nodes[nodes[cur].left].center[0], &nodes[nodes[cur].left].extent[0], dim);

				if (d <= range)
				{
					to_visit[threadid][to_visit_pos++] = nodes[cur].left;
				}

				d = 0.0;
				d = BbxToBbx(&nodes[index].center[0], &nodes[index].extent[0],
					&nodes[nodes[cur].right].center[0], &nodes[nodes[cur].right].extent[0], dim);
				if (d <= range)
				{
					to_visit[threadid][to_visit_pos++] = nodes[cur].right;
				}
			}
		}
	}
}

//sort nodes by key
void sortNode(int* gpu_level, int* gpu_idx, int num_nodes)
{
	thrust::sort_by_key(thrust::device_ptr<int>(gpu_level),
		thrust::device_ptr<int>(gpu_level + num_nodes),
		thrust::device_ptr<int>(gpu_idx));
}


__global__ void FindLevelStartAndEnd(int* levelStart,
	int* levelEnd,
	int* gpu_level,
	int* gpu_idx,
	int   num_nodes)
{
	extern __shared__ int sharedLevel[];    // blockSize + 1 elements
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int level;
	// handle case when no. of processed elements not multiple of block size
	if (index < num_nodes) {
		level = gpu_level[index];

		// Load hash data into shared memory so that we can look 
		// at neighboring element's hash value without loading
		// two hash values per thread
		sharedLevel[threadIdx.x + 1] = level;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor element hash
			sharedLevel[0] = gpu_level[index - 1];
		}
	}

	__syncthreads();

	if (index < num_nodes) {
		// If this element has a different index to the previous
		// element then it must be the first element,
		// so store the index of this element.
		// As it isn't the first element, it must also be the end of
		// the previous element's record

		if (index == 0 || level != sharedLevel[threadIdx.x])
		{
			levelStart[level] = index;
			if (index > 0)
				levelEnd[sharedLevel[threadIdx.x]] = index;
		}

		if (index == num_nodes - 1)
		{
			levelEnd[level] = index + 1;
		}
	}
}

//Phase I: Packing of queries
//We apply depth-first search(DFS) to each query point, q, to find the leaf node containing (or touching) q.
__global__ void QueriesPacking(const CUDA_AABBNode* nodes, float* queries, int num_queries, int dim, int cur, float radius, int leafstart, int* queryids, int* b_nodes, bool* touchflags, unsigned int* saddress)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		cur = 0; //trvarse from the root node

		bool flag = false; //0: contains  1: touches

		while (true) {
			if (nodes[cur].left == 0) {//cur is leaf node
				float dist = PtToBbx(&nodes[cur].cd_bnds[0], &queries[dim * idx], dim);//Distance from the query to the current leaf node
				if (dist <= (radius) * (radius))
				{
					unsigned int index = atomicAdd(saddress, 1);

					queryids[index] = idx;
					b_nodes[index] = cur;

					if (flag && !touchflags[cur - leafstart])
					{
						touchflags[cur - leafstart] = true;
					}
				}

				break;
			}
			else if (WithinBbx(&nodes[nodes[cur].left].cd_bnds[0], &queries[dim * idx], dim)) {//left child is first checked

				if (WithinBbx(&nodes[nodes[cur].right].cd_bnds[0], &queries[dim * idx], dim))
				{//this double checking is made for query points which locate in the common boundaries of two AABBs to make 
					//sure the query ponts are contained in the right leaf node
					if (WithinBbx(&nodes[nodes[nodes[cur].left].left].cd_bnds[0], &queries[dim * idx], dim) ||
						WithinBbx(&nodes[nodes[nodes[cur].left].right].cd_bnds[0], &queries[dim * idx], dim))
					{
						cur = nodes[cur].left;
					}
					else
					{
						cur = nodes[cur].right;
					}
				}
				else
				{
					cur = nodes[cur].left;
				}
			}
			else if (WithinBbx(&nodes[nodes[cur].right].cd_bnds[0], &queries[dim * idx], dim)) {//if query pint is not inside the left child, right child is checked

				cur = nodes[cur].right;
			}
			else {//if the query point is not inside the AABB, then it may touch any AABB, the nearest AABB is found
				flag = true;

				float dleft = PtToBbx(&nodes[nodes[cur].left].cd_bnds[0], &queries[dim * idx], dim);
				float dright = PtToBbx(&nodes[nodes[cur].right].cd_bnds[0], &queries[dim * idx], dim);
				if (dleft <= dright)
					cur = nodes[cur].left;
				else
					cur = nodes[cur].right;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
}

//Phase II: Bach-traversal with swept volume
//if a leaf node is touched by a query point, its searching radius is set to be 2*radius
//the points in the leaf node are first added into a list as the candidates of SRS results
//then go up traversal to find the leaf nodes overlap with the current leaf node
__global__ void BackTraversal(const CUDA_AABBNode* nodes, const int* indexes, int depth, int* leafnodes, int num_queries, bool* touchflags, int leafstart, float* pts, int num_pts, int dim, float radi, int* ret_index, int* ret_indexvsleafid, unsigned int* saddress, unsigned int* intersaddress)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		int cur = leafnodes[idx];
		int current = cur;

		if (touchflags[cur - leafstart])
			radi = 2 * radi;

		float range2 = radi * radi;

		for (int i = 0; i < nodes[cur].n_pts; i++) {
			int idxx = indexes[nodes[cur].indexes + i];

			unsigned int index = atomicAdd(intersaddress, 1);
			ret_index[index] = idxx;
			ret_indexvsleafid[index] = current;

		}

		while (nodes[cur].parent != -1) {
			// Go up
			int parent = nodes[cur].parent;
			// Search the other node
			float d = 0.0;
			if (nodes[parent].left != cur)
			{
				d = BbxToBbx(&nodes[current].center[0], &nodes[current].extent[0],
					&nodes[nodes[parent].left].center[0], &nodes[nodes[parent].left].extent[0], dim);
				if (d <= range2)
				{
					SearchAtNodeRange(threadIdx.x, current, intersaddress, nodes, indexes, depth, pts, dim, nodes[parent].left, radi, range2, ret_index, ret_indexvsleafid);
				}
			}
			else
			{
				d = BbxToBbx(&nodes[current].center[0], &nodes[current].extent[0],
					&nodes[nodes[parent].right].center[0], &nodes[nodes[parent].right].extent[0], dim);
				if (d <= range2)
				{
					SearchAtNodeRange(threadIdx.x, current, intersaddress, nodes, indexes, depth, pts, dim, nodes[parent].right, radi, range2, ret_index, ret_indexvsleafid);
				}
			}

			cur = parent;
		}
		idx = atomicAdd(saddress, 1);
	}
}

__global__ void BackTraversalByDataPoints(const CUDA_AABBNode* nodes, const int* indexes, int depth, int* leafnodes, int num_queries, int leafstart, float* pts, int num_pts, int dim, float radi, int* ret_index, int* ret_indexvsleafid, unsigned int* saddress, unsigned int* intersaddress)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		int cur = leafnodes[idx];
		int current = cur;

		float range2 = radi * radi;

		for (int i = 0; i < nodes[cur].n_pts; i++) {
			int idxx = indexes[nodes[cur].indexes + i];

			unsigned int index = atomicAdd(intersaddress, 1);
			ret_index[index] = idxx;
			ret_indexvsleafid[index] = current;

		}

		while (nodes[cur].parent != -1) {
			// Go up
			int parent = nodes[cur].parent;
			// Search the other node
			float d = 0.0;
			if (nodes[parent].left != cur)
			{
				d = BbxToBbx(&nodes[current].center[0], &nodes[current].extent[0],
					&nodes[nodes[parent].left].center[0], &nodes[nodes[parent].left].extent[0], dim);
				if (d <= range2)
				{
					SearchAtNodeRange(threadIdx.x, current, intersaddress, nodes, indexes, depth, pts, dim, nodes[parent].left, radi, range2, ret_index, ret_indexvsleafid);
				}
			}
			else
			{
				d = BbxToBbx(&nodes[current].center[0], &nodes[current].extent[0],
					&nodes[nodes[parent].right].center[0], &nodes[nodes[parent].right].extent[0], dim);
				if (d <= range2)
				{
					SearchAtNodeRange(threadIdx.x, current, intersaddress, nodes, indexes, depth, pts, dim, nodes[parent].right, radi, range2, ret_index, ret_indexvsleafid);
				}
			}

			cur = parent;
		}
		idx = atomicAdd(saddress, 1);
	}
}

__device__ void SearchAtNode_NN(int idx, const CUDA_AABBNode* nodes, const int* indexes, float* pts, float* queries, int cur, int* b_index, float* b_dist, int* b_node)
{
	// Finds the first potential candidate
	int best_idx = 0;
	float best_dist = FLT_MAX;

	while (true) {
		if (nodes[cur].left == 0) {
			*b_node = cur;

			for (int i = 0; i < nodes[cur].n_pts; i++)
			{
				int idxx = indexes[nodes[cur].indexes + i];
				float dist = PtToPt(queries, &pts[DIMENSIONS * idxx], DIMENSIONS);

				if (dist > 0.000001 & dist < best_dist)
				{
					best_dist = dist;
					best_idx = idxx;
				}
			}

			break;
		}
		else if (WithinBbx(&nodes[nodes[cur].left].cd_bnds[0], queries, DIMENSIONS)) {
			cur = nodes[cur].left;
		}
		else {
			cur = nodes[cur].right;
		}
	}

	*b_index = best_idx;
	*b_dist = best_dist;
}

__device__ void SearchAtNodeRange_NN(int idx, const CUDA_AABBNode* nodes, const int* indexes, float* pts, float* queries, int cur, float range, int* r_index, float* r_dist)
{
	// Goes through all the nodes that are within "range"
	int best_idx = 0;
	float best_dist = FLT_MAX;

	// Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
	// We'll use a fixed length stack, increase this as required
	int to_visit[CUDA_STACK];
	int to_visit_pos = 0;

	to_visit[to_visit_pos++] = cur;

	while (to_visit_pos) {
		int next_search[CUDA_STACK];
		int next_search_pos = 0;

		while (to_visit_pos) {
			cur = to_visit[to_visit_pos - 1];
			to_visit_pos--;

			if (nodes[cur].left == 0) {
				for (int i = 0; i < nodes[cur].n_pts; i++) {
					int idxx = indexes[nodes[cur].indexes + i];
					float d = PtToPt(queries, &pts[DIMENSIONS * idxx], DIMENSIONS);

					if (d < best_dist)
					{
						best_dist = d;
						best_idx = idxx;
					}
				}
			}
			else {
				if (PtToBbx(&nodes[nodes[cur].left].cd_bnds[0], queries, DIMENSIONS) <= range * range)
				{
					next_search[next_search_pos++] = nodes[cur].left;
				}
				if (PtToBbx(&nodes[nodes[cur].right].cd_bnds[0], queries, DIMENSIONS) <= range * range)
				{
					next_search[next_search_pos++] = nodes[cur].right;
				}
			}
		}
		for (int i = 0; i < next_search_pos; i++)
			to_visit[i] = next_search[i];

		to_visit_pos = next_search_pos;
	}

	*r_index = best_idx;
	*r_dist = best_dist;
}


__device__ void Search_NN(int index, const CUDA_AABBNode* nodes, const int* indexes, float* pts, float* queries, float radius, int* ret_ind, float* ret_dis)
{
	// Find the first closest node, this will be the upper bound for the next searches
	int best_node = 0;
	int best_idx = 0;
	float best_dist = FLT_MAX;


	SearchAtNode_NN(index, nodes, indexes, pts, queries, 0 /* root */, &best_idx, &best_dist, &best_node);

	// Now find other possible candidates
	int cur = best_node;

	while (nodes[cur].parent != -1) {
		// Go up
		int parent = nodes[cur].parent;


		// Search the other node
		float tmp_dist = FLT_MAX;
		int tmp_idx;

		// Search opposite node
		if (nodes[parent].left != cur)
		{
			if (PtToBbx(&nodes[nodes[parent].left].cd_bnds[0], queries, DIMENSIONS) <= radius * radius)
			{
				SearchAtNodeRange_NN(index, nodes, indexes, pts, queries, nodes[parent].left, radius, &tmp_idx, &tmp_dist);
			}
		}
		else
		{
			if (PtToBbx(&nodes[nodes[parent].right].cd_bnds[0], queries, DIMENSIONS) <= radius * radius)
			{
				SearchAtNodeRange_NN(index, nodes, indexes, pts, queries, nodes[parent].right, radius, &tmp_idx, &tmp_dist);
			}
		}

		if (tmp_dist < best_dist) {
			best_dist = tmp_dist;
			best_idx = tmp_idx;
		}

		cur = parent;
	}

	if (best_dist <= radius * radius)
	{
		*ret_ind = best_idx;
		*ret_dis = best_dist;
	}
	else
	{
		*ret_ind = -1;
		*ret_dis = -1.0;
	}
}

__global__ void SearchBatch(const CUDA_AABBNode* nodes, const int* indexes, float* pts, int num_pts, float* queries, int num_queries, float radius, int* ret_index, float* ret_dist)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		float queryPt[DIMENSIONS];
		for (int i = 0; i < DIMENSIONS; i++) {
			queryPt[i] = queries[DIMENSIONS * idx + i];
		}
		Search_NN(idx, nodes, indexes, pts, queryPt, radius, &(ret_index[idx]), &(ret_dist[idx]));

		idx += blockDim.x * gridDim.x;
	}
}



__global__ void FindLeafStartAndEnd(int* leafStart,
	int* leafEnd,
	int num_leafs,
	int* gpu_level,
	int* gpu_idx,
	int   num_nodes)
{
	extern __shared__ int sharedLevel[];    // blockSize + 1 elements
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int level;
	// handle case when no. of queries not multiple of block size
	if (index < num_nodes) {
		level = gpu_level[index];

		// Load hash data into shared memory so that we can look 
		// at neighboring query's hash value without loading
		// two hash values per thread
		sharedLevel[threadIdx.x + 1] = level;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor query hash
			sharedLevel[0] = gpu_level[index - 1];
		}
	}

	__syncthreads();

	if (index < num_nodes) {
		// If this query has a different leaf node index to the previous
		// query then it must be the first query in the leaf node,
		// so store the index of this query in the leaf node.
		// As it isn't the first query, it must also be the leaf node end of
		// the previous query's leaf node

		if (index == 0 || level != sharedLevel[threadIdx.x])
		{
			leafStart[level - num_leafs + 1] = index;
			if (index > 0)
				leafEnd[sharedLevel[threadIdx.x] - num_leafs + 1] = index;
		}

		if (index == num_nodes - 1)
		{
			leafEnd[level - num_leafs + 1] = index + 1;
		}
	}


}

//Phase III: Result Compaction
__global__ void kernel_CalculateDistance(const CUDA_AABBNode* nodes, const int* indexes, float* pts, float* queries, int num_queries, int* leafstart, int* leafend, int l_start, int* searchnodevsqueryids, int dim, int* searchpts, int* searchptvsleafids, float radius, int* ret_index, int* ret_index_queryids, unsigned int* intersaddress, unsigned int* saddress)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		int id = searchpts[idx];

		float r = radius * radius;
		int r_leaf = searchptvsleafids[idx];

		int s = leafstart[r_leaf - l_start];
		int e = leafend[r_leaf - l_start];

		int count = e - s;

		for (int i = 0; i < count; i++) {
			int idxx = searchnodevsqueryids[s + i];
			float d = PtToPt(&pts[dim * id], &queries[dim * idxx], dim);

			if (d <= r)
			{
				unsigned int index = atomicAdd(intersaddress, 1);

				ret_index[index] = id;
				ret_index_queryids[index] = idxx;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_CalculateDistanceByDataPoints(const CUDA_AABBNode* nodes, const int* indexes, float* pts,
	float* queries, int num_queries, int dim, int* searchpts,
	int* searchptvsleafids, float radius, int* ret_index, int* ret_index_queryids,
	unsigned int* intersaddress, unsigned int* saddress)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		int id = searchpts[idx];

		float r = radius * radius;
		int r_leaf = searchptvsleafids[idx];

		for (int i = 0; i < nodes[r_leaf].n_pts; i++) {
			int idxx = indexes[nodes[r_leaf].indexes + i];
			float d = PtToPt(&pts[dim * id], &queries[dim * idxx], dim);

			if (d <= r)
			{
				unsigned int index = atomicAdd(intersaddress, 1);

				ret_index[index] = id;
				ret_index_queryids[index] = idxx;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
}
//output the result in a required form
__global__ void kernel_output2(int num_queries, float* pts, float* queries, int dim, int* start, int* end, int* ptsIndex, int* ret_indexes, int maxNeigh, float* ret_dists, int* n_neigh, unsigned int* sadd)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_queries)
	{
		int s = start[idx];
		int e = end[idx];
		int count = e - s;

		int loc = 0;
		for (int i = 0; i < count; i++)
		{
			if (loc < maxNeigh)
			{
				int index = ptsIndex[s + i];

				float d = PtToPt(&queries[dim * idx], &pts[dim * index], dim);

				if (d > 0.000001)
				{
					ret_indexes[idx * maxNeigh + loc] = index;
					ret_dists[idx * maxNeigh + loc] = d;
					loc++;
				}
			}
		}
		n_neigh[idx] = loc;

		idx += blockDim.x * gridDim.x;
		//idx = atomicAdd(sadd,1);
	}
}

//Coherent SRS
bool SRStree_GPU::Search(float* gpu_queries, int query_num, float radii, int max_SearchNeigh, int* real_gpu_ret_indexes, float* real_gpu_ret_dist, int* real_gpu_neigh)
{
	int threads = THREAD_PER_BLOCK;
	int blocks = NUMBER_OF_BLOCK;

	int MaxSearchNeigh = max_SearchNeigh * 3;
	int leafnode_start = m_num_nodes - m_num_leafnodes;

	unsigned int* sadd;
	unsigned int value = (32 * 256 - 1);
	unsigned int* sadd2;
	unsigned int* sadd3;
	unsigned int* intersadd;
	cudaMalloc((void**)&intersadd, sizeof(unsigned int));
	cudaMemset(intersadd, 0, sizeof(unsigned int));

	cudaMalloc((void**)&sadd, sizeof(unsigned int));
	cudaMalloc((void**)&sadd2, sizeof(unsigned int));
	cudaMalloc((void**)&sadd3, sizeof(unsigned int));
	cudaMemcpy(sadd, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(sadd2, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(sadd3, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);


	//Phase I: Packing of Queries
	int* searchNodes;
	cudaMalloc((void**)&searchNodes, sizeof(int) * query_num);
	cudaMemset(searchNodes, 0xffffffff, sizeof(int) * query_num);
	int* searchNodevsqueryids;
	cudaMalloc((void**)&searchNodevsqueryids, sizeof(int) * query_num);
	bool* touch_flags;
	cudaMalloc((void**)&touch_flags, sizeof(bool) * m_num_leafnodes);
	cudaMemset(touch_flags, 0, sizeof(bool) * m_num_leafnodes);

	QueriesPacking << <blocks, threads >> > (m_gpu_nodes, gpu_queries, query_num, dimension, 0, radii, leafnode_start, searchNodevsqueryids, searchNodes, touch_flags, intersadd);

	unsigned int* inter_num = new unsigned int[1];
	cudaMemcpy(inter_num, intersadd, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int sum_packedqueries = (*inter_num);
#ifdef INFORMATION_PRINT
	printf("\nsum_packedqueries %d\n", sum_packedqueries);
#endif

	//sort nodes keyed by the IDs of leaf nodes
	sortNode(searchNodes, searchNodevsqueryids, sum_packedqueries);
	int* leaf_start;
	cudaMalloc((void**)&leaf_start, sizeof(int) * m_num_leafnodes);
	int* leaf_end;
	cudaMalloc((void**)&leaf_end, sizeof(int) * m_num_leafnodes);
	int numThreads, numBlocks;
	computeGridSize(sum_packedqueries, 256, numBlocks, numThreads);
	cudaMemset(leaf_start, 0xffffffff, m_num_leafnodes * sizeof(int));
	cudaMemset(leaf_end, 0xffffffff, m_num_leafnodes * sizeof(int));
	int smemSize = sizeof(int) * (numThreads + 1);
	FindLeafStartAndEnd << <numBlocks, numThreads, smemSize >> > (leaf_start, leaf_end, m_num_leafnodes, searchNodes, searchNodevsqueryids, sum_packedqueries);

	//compact "searchNodes" into a reduced list "involvedNodes" by removing the repeated entries
	int* h_searchNodes = new int[sum_packedqueries];
	cudaMemcpy(h_searchNodes, searchNodes, sum_packedqueries * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(searchNodes);
	int* output_end = thrust::unique(h_searchNodes,
		(h_searchNodes + sum_packedqueries));
	int num_involvedleafnodes = output_end - h_searchNodes;
#ifdef INFORMATION_PRINT
	printf("involvedleafnodes %d\n", num_involvedleafnodes);
#endif
	int* involvedNodes;
	cudaMalloc((void**)&involvedNodes, sizeof(int) * num_involvedleafnodes);
	cudaMemcpy(involvedNodes, h_searchNodes, sizeof(int) * num_involvedleafnodes, cudaMemcpyHostToDevice);
	delete[] h_searchNodes;

	//Phase II: Back-traversal with swept volume
	int* gpu_ret_indexes;
	int* gpu_ret_indexvsleafid;
	cudaMalloc((void**)&gpu_ret_indexes, sizeof(int) * num_involvedleafnodes * MaxSearchNeigh);
	cudaMalloc((void**)&gpu_ret_indexvsleafid, sizeof(int) * num_involvedleafnodes * MaxSearchNeigh);
	cudaMemset(gpu_ret_indexes, 0xffffffff, num_involvedleafnodes * MaxSearchNeigh * sizeof(int));
	cudaMemset(gpu_ret_indexvsleafid, 0xffffffff, num_involvedleafnodes * MaxSearchNeigh * sizeof(int));
	//int max_SearchNeigh = MaxSearchNeigh/3;
	cudaMemset(intersadd, 0, sizeof(unsigned int));

	BackTraversal << <blocks, threads >> > (m_gpu_nodes, m_gpu_indexes, m_max_level, involvedNodes, num_involvedleafnodes, touch_flags, leafnode_start, m_gpu_points, m_num_points, dimension, radii, gpu_ret_indexes, gpu_ret_indexvsleafid, sadd, intersadd);

	cudaThreadSynchronize();
	cudaFree(touch_flags);
	cudaFree(involvedNodes);
	unsigned int* inter_num2 = new unsigned int[1];
	cudaMemcpy(inter_num2, intersadd, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if ((*inter_num2) >= (unsigned int)(num_involvedleafnodes * MaxSearchNeigh))
	{
		printf("memory usage overflow!\n");
		return false;
	}
#ifdef INFORMATION_PRINT
	printf("inter_num2 %u\n", (*inter_num2));
#endif

	size_t free_byte;
	size_t total_byte;
	cudaError_t	cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}
	double free_db = (double)free_byte;
#ifdef INFORMATION_PRINT
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
#endif

	//Phase III: Result compaction
	unsigned int sum = (*inter_num2);
	unsigned int size = (unsigned int)(0.6 * free_db / 4.0 / 2.0 / (double)sum);
#ifdef INFORMATION_PRINT
	printf("size %u\n", size);
#endif
	cudaMemset(intersadd, 0, sizeof(unsigned int));
	int* temp_temp_gpu_ret_indexes;
	cudaMalloc((void**)&temp_temp_gpu_ret_indexes, sizeof(int) * sum * size);
	cudaMemset(temp_temp_gpu_ret_indexes, 0xffffffff, (sum * size) * sizeof(int));
	int* temp_temp_gpu_ret_index_queryids;
	cudaMalloc((void**)&temp_temp_gpu_ret_index_queryids, sizeof(int) * (sum * size));
	cudaMemset(temp_temp_gpu_ret_index_queryids, 0xffffffff, (sum * size) * sizeof(int));

	kernel_CalculateDistance << <blocks, threads >> > (m_gpu_nodes, m_gpu_indexes, m_gpu_points, gpu_queries, sum, leaf_start, leaf_end, leafnode_start, searchNodevsqueryids, dimension, gpu_ret_indexes, gpu_ret_indexvsleafid, radii, temp_temp_gpu_ret_indexes, temp_temp_gpu_ret_index_queryids, intersadd, sadd3);

	cudaFree(gpu_ret_indexes);
	cudaFree(gpu_ret_indexvsleafid);
	cudaFree(leaf_start);
	cudaFree(leaf_end);
	cudaFree(searchNodevsqueryids);
	unsigned int* sumpt = new unsigned int[1];
	cudaMemcpy(sumpt, intersadd, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int sumpts = (*sumpt);
	if (sumpts >= sum * size)
	{
		printf("memory usage2 overflow!\n");
	}
	else
	{
#ifdef INFORMATION_PRINT
		printf("sumpts %u\n", sumpts);
#endif
	}
	sortNode(temp_temp_gpu_ret_index_queryids, temp_temp_gpu_ret_indexes, sumpts);
	int* query_start;
	cudaMalloc((void**)&query_start, sizeof(int) * query_num);
	int* query_end;
	cudaMalloc((void**)&query_end, sizeof(int) * query_num);
	computeGridSize(sumpts, 256, numBlocks, numThreads);
	cudaMemset(query_start, 0xffffffff, query_num * sizeof(int));
	cudaMemset(query_end, 0xffffffff, query_num * sizeof(int));
	smemSize = sizeof(int) * (numThreads + 1);
	FindLevelStartAndEnd << <numBlocks, numThreads, smemSize >> > (query_start, query_end, temp_temp_gpu_ret_index_queryids, temp_temp_gpu_ret_indexes, sumpts);
	cudaFree(temp_temp_gpu_ret_index_queryids);

	//output the result in requried form
	kernel_output2 << <blocks, threads >> > (query_num, m_gpu_points, gpu_queries, dimension, query_start, query_end, temp_temp_gpu_ret_indexes, real_gpu_ret_indexes, max_SearchNeigh, real_gpu_ret_dist, real_gpu_neigh, sadd2);

	cudaFree(query_start);
	cudaFree(query_end);
	cudaFree(temp_temp_gpu_ret_indexes);
	cudaFree(sadd);
	cudaFree(intersadd);
	cudaFree(sadd2);
	cudaFree(sadd3);

	return true;
}

bool SRStree_GPU::SearchByDataPoints(float radii, int max_SearchNeigh,
	int* real_gpu_ret_indexes, float* real_gpu_ret_dist, int* real_gpu_neigh)
{
	int threads = THREAD_PER_BLOCK;
	int blocks = NUMBER_OF_BLOCK;

	int MaxSearchNeigh = max_SearchNeigh * 3;
	int leafnode_start = m_num_nodes - m_num_leafnodes;

	float* gpu_queries = m_gpu_points;	// Query by data-points
	int query_num = m_num_points;

	unsigned int* sadd;
	unsigned int value = (32 * 256 - 1);
	unsigned int* sadd2;
	unsigned int* sadd3;
	unsigned int* intersadd;
	cudaMalloc((void**)&intersadd, sizeof(unsigned int));
	cudaMemset(intersadd, 0, sizeof(unsigned int));

	cudaMalloc((void**)&sadd, sizeof(unsigned int));
	cudaMalloc((void**)&sadd2, sizeof(unsigned int));
	cudaMalloc((void**)&sadd3, sizeof(unsigned int));
	cudaMemcpy(sadd, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(sadd2, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(sadd3, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);

	//Phase I: Packing of Queries-----bypass

	//Phase II: Back-traversal with swept volume
	int* gpu_ret_indexes;
	int* gpu_ret_indexvsleafid;
	cudaMalloc((void**)&gpu_ret_indexes, sizeof(int) * m_num_leafnodes * MaxSearchNeigh);
	cudaMalloc((void**)&gpu_ret_indexvsleafid, sizeof(int) * m_num_leafnodes * MaxSearchNeigh);
	cudaMemset(gpu_ret_indexes, 0xffffffff, m_num_leafnodes * MaxSearchNeigh * sizeof(int));
	cudaMemset(gpu_ret_indexvsleafid, 0xffffffff, m_num_leafnodes * MaxSearchNeigh * sizeof(int));
	//int max_SearchNeigh = MaxSearchNeigh/3;
	cudaMemset(intersadd, 0, sizeof(unsigned int));
	//---------------------------------------------------------------------------------------
	BackTraversalByDataPoints << <blocks, threads >> > (m_gpu_nodes, m_gpu_indexes, m_max_level, m_gpu_leafnodes,
		m_num_leafnodes, leafnode_start, m_gpu_points, m_num_points, dimension, radii, gpu_ret_indexes, gpu_ret_indexvsleafid, sadd, intersadd);
	//---------------------------------------------------------------------------------------
	cudaThreadSynchronize();

	unsigned int* inter_num2 = new unsigned int[1];
	cudaMemcpy(inter_num2, intersadd, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if ((*inter_num2) >= (unsigned int)(m_num_leafnodes * MaxSearchNeigh)) {
		printf("memory usage overflow!\n");
		return false;
	}
#ifdef INFORMATION_PRINT
	printf("inter_num2 %u\n", (*inter_num2));
#endif

	size_t free_byte;
	size_t total_byte;
	cudaError_t	cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}
	double free_db = (double)free_byte;
#ifdef INFORMATION_PRINT
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
#endif

	//Phase III: Result compaction
	unsigned int sum = (*inter_num2);
	unsigned int size = (unsigned int)(0.6 * free_db / 4.0 / 2.0 / (double)sum);
#ifdef INFORMATION_PRINT
	printf("size %u\n", size);
#endif
	cudaMemset(intersadd, 0, sizeof(unsigned int));
	int* temp_temp_gpu_ret_indexes;
	cudaMalloc((void**)&temp_temp_gpu_ret_indexes, sizeof(int) * sum * size);
	cudaMemset(temp_temp_gpu_ret_indexes, 0xffffffff, (sum * size) * sizeof(int));
	int* temp_temp_gpu_ret_index_queryids;
	cudaMalloc((void**)&temp_temp_gpu_ret_index_queryids, sizeof(int) * (sum * size));
	cudaMemset(temp_temp_gpu_ret_index_queryids, 0xffffffff, (sum * size) * sizeof(int));
	//---------------------------------------------------------------------------------------
	kernel_CalculateDistanceByDataPoints << <blocks, threads >> > (m_gpu_nodes, m_gpu_indexes, m_gpu_points,
		gpu_queries, sum, dimension, gpu_ret_indexes, gpu_ret_indexvsleafid, radii,
		temp_temp_gpu_ret_indexes, temp_temp_gpu_ret_index_queryids, intersadd, sadd3);
	//---------------------------------------------------------------------------------------
	cudaFree(gpu_ret_indexes);
	cudaFree(gpu_ret_indexvsleafid);

	unsigned int* sumpt = new unsigned int[1];
	cudaMemcpy(sumpt, intersadd, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int sumpts = (*sumpt);
	if (sumpts >= sum * size)
	{
		printf("memory usage2 overflow!\n");
	}
	else
	{
#ifdef INFORMATION_PRINT
		printf("sumpts %u\n", sumpts);
#endif
	}
	sortNode(temp_temp_gpu_ret_index_queryids, temp_temp_gpu_ret_indexes, sumpts);
	int* query_start;
	cudaMalloc((void**)&query_start, sizeof(int) * query_num);
	int* query_end;
	cudaMalloc((void**)&query_end, sizeof(int) * query_num);
	int numThreads, numBlocks;
	computeGridSize(sumpts, 256, numBlocks, numThreads);
	cudaMemset(query_start, 0xffffffff, query_num * sizeof(int));
	cudaMemset(query_end, 0xffffffff, query_num * sizeof(int));
	int smemSize = sizeof(int) * (numThreads + 1);
	FindLevelStartAndEnd << <numBlocks, numThreads, smemSize >> > (query_start, query_end, temp_temp_gpu_ret_index_queryids, temp_temp_gpu_ret_indexes, sumpts);
	cudaFree(temp_temp_gpu_ret_index_queryids);

	//output the result in requried form
	kernel_output2 << <blocks, threads >> > (query_num, m_gpu_points, gpu_queries, dimension, query_start, query_end, temp_temp_gpu_ret_indexes, real_gpu_ret_indexes, max_SearchNeigh, real_gpu_ret_dist, real_gpu_neigh, sadd2);

	cudaFree(query_start);
	cudaFree(query_end);
	cudaFree(temp_temp_gpu_ret_indexes);
	cudaFree(sadd);
	cudaFree(intersadd);
	cudaFree(sadd2);
	cudaFree(sadd3);

	return true;

}

bool SRStree_GPU::SearchNN(float* gpu_queries, int query_num, float radii, int* real_gpu_ret_index, float* real_gpu_ret_dist)
{
	int threads = THREAD_PER_BLOCK;
	int blocks = NUMBER_OF_BLOCK;

	SearchBatch << <blocks, threads >> > (m_gpu_nodes, m_gpu_indexes, m_gpu_points, m_num_points, gpu_queries, query_num, radii, real_gpu_ret_index, real_gpu_ret_dist);

	return true;
}


//update leaf nodes
__global__ void Update_AABB_Leafnode(int* leafnodes, int num_leaf, CUDA_AABBNode* nodes, const int* indexes, float* pts, int dim, unsigned int* saddress, float* volVar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_leaf)
	{
		int nodeid = leafnodes[idx];
		int n_pts = nodes[nodeid].n_pts;

		float vol = 1.0;
		for (int d = 0; d < dim; d++)
		{
			float min = pts[dim * indexes[nodes[nodeid].indexes] + d];						// compute max and min coords
			float max = pts[dim * indexes[nodes[nodeid].indexes] + d];
			for (int i = 1; i < n_pts; i++)
			{
				float c = pts[dim * indexes[nodes[nodeid].indexes + i] + d];
				if (c < min) min = c;
				else if (c > max) max = c;
			}

			nodes[nodeid].cd_bnds[d] = min;
			nodes[nodeid].cd_bnds[dim + d] = max;

			nodes[nodeid].center[d] = 0.5 * (nodes[nodeid].cd_bnds[d] + nodes[nodeid].cd_bnds[d + dim]);
			nodes[nodeid].extent[d] = 0.5 * (nodes[nodeid].cd_bnds[d + dim] - nodes[nodeid].cd_bnds[d]);

			vol *= nodes[nodeid].cd_bnds[d + dim] - nodes[nodeid].cd_bnds[d];
		}

		float ratio = -1.0;
		if (nodes[nodeid].volume > 0.000001)
		{
			ratio = vol / nodes[nodeid].volume;
		}
		volVar[nodeid] = ratio;

		idx += blockDim.x * gridDim.x;
		//idx = atomicAdd(saddress,1);
	}
}

//update split nodes
__global__ void Update_AABB_splitnode(CUDA_AABBNode* nodes, int* gpu_start, int* gpu_end, int* gpu_idx, int level, int num_nodes, int dim, unsigned int* saddress, float* volVar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_nodes)
	{
		int index = gpu_start[level] + idx;
		if (index < gpu_end[level])
		{
			int nodeid = gpu_idx[index];
			if (nodes[nodeid].type == 0)
			{
				int leftid = nodes[nodeid].left;
				int rightid = nodes[nodeid].right;

				//float min[4],max[4];
				float vol = 1.0;
				for (int i = 0; i < dim; i++)
				{
					float min, max;
					if (nodes[leftid].cd_bnds[i] < nodes[rightid].cd_bnds[i])
						min = nodes[leftid].cd_bnds[i];
					else
						min = nodes[rightid].cd_bnds[i];

					if (nodes[leftid].cd_bnds[dim + i] < nodes[rightid].cd_bnds[dim + i])
						max = nodes[rightid].cd_bnds[dim + i];
					else
						max = nodes[leftid].cd_bnds[dim + i];

					nodes[nodeid].cd_bnds[i] = min;
					nodes[nodeid].cd_bnds[dim + i] = max;

					nodes[nodeid].center[i] = 0.5 * (nodes[nodeid].cd_bnds[i] + nodes[nodeid].cd_bnds[i + dim]);
					nodes[nodeid].extent[i] = 0.5 * (nodes[nodeid].cd_bnds[i + dim] - nodes[nodeid].cd_bnds[i]);

					vol *= nodes[nodeid].cd_bnds[i + dim] - nodes[nodeid].cd_bnds[i];
				}

				float ratio = -1.0;
				if (nodes[nodeid].volume > 0.000001)
				{
					ratio = vol / nodes[nodeid].volume;
				}
				volVar[nodeid] = ratio;
			}
		}

		idx += blockDim.x * gridDim.x;
		//idx = atomicAdd(saddress,1);
	}
}

//calculate level for each node
__global__ void calcLevelForEachNode(CUDA_AABBNode* nodes, int num_nodes, int* gpu_level, int* gpu_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (idx < num_nodes)
	{
		gpu_level[idx] = nodes[idx].level;
		gpu_idx[idx] = idx;

		idx += blockDim.x * gridDim.x;
	}
}

//Hierarchical updating
void SRStree_GPU::RefitAABB()
{
	unsigned int* saddr;
	unsigned int value = 1 * (32 * 256 - 1);

	cudaMalloc((void**)&saddr, sizeof(unsigned int));
	cudaMemcpy(saddr, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);

	float* nodeVolumeVariation;
	cudaMalloc((void**)&nodeVolumeVariation, sizeof(float) * m_num_nodes);

	int threads = THREAD_PER_BLOCK;
	int blocks = NUMBER_OF_BLOCK;

	Update_AABB_Leafnode << <blocks, threads >> > (m_gpu_leafnodes, m_num_leafnodes, m_gpu_nodes, m_gpu_indexes, m_gpu_points, dimension, saddr, nodeVolumeVariation);

	int Level = m_max_level - 1;
	while (Level != -1)
	{
		float temp = pow(2.0f, (float)Level);
		int n_nodes = (int)(temp);

		Update_AABB_splitnode << <blocks, threads >> > (m_gpu_nodes, m_gpu_start, m_gpu_end, m_gpu_idx, Level, n_nodes, dimension, saddr, nodeVolumeVariation);

		Level--;
	}

	cudaThreadSynchronize();

	float* h_nodeVolumeVariation = new float[m_num_nodes];
	cudaMemcpy(h_nodeVolumeVariation, nodeVolumeVariation, sizeof(float) * m_num_nodes, cudaMemcpyDeviceToHost);

	float* result = thrust::max_element(h_nodeVolumeVariation
		, h_nodeVolumeVariation + m_num_nodes);

	cudaFree(nodeVolumeVariation);
	delete h_nodeVolumeVariation;

	if ((*result) > volThrd)
	{
		flag_rebuild = true;
	}

	cudaFree(saddr);
}

void SRStree_GPU::SortNodes()
{
	int threads = THREAD_PER_BLOCK;
	int blocks = NUMBER_OF_BLOCK;

	calcLevelForEachNode << <blocks, threads >> > (m_gpu_nodes, m_num_nodes, m_gpu_level, m_gpu_idx);

	int numThreads, numBlocks;
	computeGridSize(m_num_nodes, 256, numBlocks, numThreads);

	cudaMemset(m_gpu_start, 0xffffffff, (m_max_level + 1) * sizeof(int));
	int smemSize = sizeof(int) * (numThreads + 1);
	FindLevelStartAndEnd << <numBlocks, numThreads, smemSize >> > (m_gpu_start, m_gpu_end, m_gpu_level, m_gpu_idx, m_num_nodes);
}

void SRStree_GPU::SetDataPoints(float* pos, int num)
{
	cudaMemcpy(m_gpu_points, pos, sizeof(float) * num * dimension, cudaMemcpyHostToDevice);
}

bool SRStree_GPU::isRebuildNeeded()
{
	return flag_rebuild;
}