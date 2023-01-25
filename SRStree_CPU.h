#ifndef __SRSTREE_CPU_H__
#define __SRSTREE_CPU_H__

#include <vector>

#include "SRStree.h"


using namespace std;

//----------------------------------------------------------------------
//	Uniform AABB-tree node in CPU
//
//		Nodes of the AABB-tree in CPU store the type, index (for GPU use), level, bounding box,
//		pointers of its parent, left child and right child, index of its parent, 
//		left child and right child (for GPU use), number of points contained in bucket and point
//		indexes. These information are used for the construction of the final 1D array in GPU.
//
//      After the tree is constructed, all the nodes of the tree are reorganized and stored
//      according to the breath-first traversal (BFT) of the tree. 
//----------------------------------------------------------------------
class AABBNode_CPU
{
public:
	AABBNode_CPU()			//constructor
	{
		type = -1;
		parent = NULL;
		left = NULL;
		right = NULL;
		id = -1;
		level = -1;
		_parent = -1;
		_left = -1;
		_right = -1;
		n_pts = -1;
	}
	~AABBNode_CPU()			//destructor
	{
		/*if (parent!= NULL)
			delete parent;
		if (left!= NULL)
			delete left;
		if (right!= NULL)
			delete right;*/
	}

	int type;		//leaf or split node 0:split 1:leaf
	int id;			// for GPU
	int level;
	AABBNode_CPU* parent, * left, * right;
	int _parent, _left, _right; // for GPU
	vector<float> bnd_box;		// bounding box
	int n_pts;			// no. points in bucket
	vector <int> bkt; // index to points
};

class SRStree_CPU
{
public:
	SRStree_CPU();
	~SRStree_CPU();

	void Reconstruct(SRStree* aabbtree);
	int GetNumNodes() const { return m_id; }
	int GetNumLeafs() const { return m_num_leafnodes; }
	AABBNode_CPU* GetRoot() const { return m_root; }
	int GetDim() const { return dim; }
	int GetMaxLevel() { return max_level; }

public:
	int dim;				// dimension of space
	int n_pts;				// number of points in tree
	int	bkt_size;			// bucket size
	AABBNode_CPU* m_root;				// root of AABB-tree


	int m_id; // current node ID
	int max_level;	// maximum level of the tree
	int m_num_leafnodes;	// no. of leaf nodes
};


#endif
