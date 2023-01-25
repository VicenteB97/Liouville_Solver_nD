#ifndef __SRSTREE_H__
#define __SRSTREE_H__

#include <vector>
#include "Global_Declarations.cuh"

//#define INFORMATION_PRINT	true		// Disable this macro will block the usage of printf in computation

using namespace std;

//----------------------------------------------------------------------
//	Generic AABB-tree node
//
//		Nodes in AABB-trees are of two types, splitting nodes which contain
//		splitting information (bounding box) and leaf nodes which contain point
//		information (an array of points stored in a bucket).  This is
//		handled by making a generic class AABB_node, which is essentially an
//		empty shell, and then deriving the leaf and splitting nodes from
//		this.
//----------------------------------------------------------------------
class AABB_node                  		// node for AABB-tree
{
public:
	~AABB_node() { }					// destructor (none)

	virtual void print() = 0;           // print node

};

//----------------------------------------------------------------------
//	AABB-tree:
//		Construction:
//		-------------
//		The constructor is given the point array, number of points,
//		dimension, bucket size (default = 1). The point array is not copied, and
//		is assumed to be kept constant throughout the lifetime of the structure.
//
//		The tree is dynamically allocated in the constructor and 
//      is given as a pointer to its root node
//		(root).  These nodes are automatically deallocated when the tree
//		is deleted.
//
//		Each leaf of the tree does not contain a pointer directly to a
//		point, but rather contains a pointer to a "bucket", which is an
//		array consisting of point indices. 
//
//		The tree consists of a number of information. These consist of the following:
//
//		dim						Dimension of space
//		n_pts					Number of points currently in the tree
//		bkt_size				Maximum bucket size (no. of points per leaf)
//		bnd_box_lo				Bounding box low point
//		bnd_box_hi				Bounding box high point
//
//----------------------------------------------------------------------

class SRStree
{
public:
	SRStree();
	SRStree(float* pa,							// build from point array
		int				n,				// number of points
		int				dd,				// dimension
		int				bs = 1);			// bucket size
	~SRStree();

	AABB_node* GetRoot() const { return root; }
	int GetDim() const { return dim; }

	void EnclRect(				// compute smallest enclosing rectangle
		float* pa,				// point array
		int* pidx,			// point indices
		int						n,				// number of points
		int						dim,			// dimension
		float* bnds_lo,
		float* bnds_hi);
	float aabbSpread(				// compute point spread along dimension
		float* pa,				// point array
		int* pidx,			// point indices
		int					n,				// number of points
		int          dim,
		int					d);				// dimension to check
	int aabbMaxSpread(				// compute point spread along dimension
		float* pa,				// point array
		int* pidx,			// point indices
		int					n,				// number of points
		int					d);				// dimension to check
	void aabbMinMax(					// compute min and max coordinates along dim
		float* pa,				// point array
		int* pidx,			// point indices
		int					n,				// number of points
		int          dim,
		int					d,				// dimension to check
		float& min,			// minimum value (returned)
		float& max);			// maximum value (returned)

	void aabbPlaneSplit(				// split points by a plane
		float* pa,				// points to split
		int* pidx,			// point indices
		int					n,				// number of points
		int          dim,
		int					d,				// dimension along which to split
		float			cv,				// cutting value
		int& br1,			// first break (values < cv)
		int& br2);			// second break (values == cv)

	void median_split(
		float* pa,				// point array
		int* pidx,			// point indices (permuted on return)
		float* bnds,			// bounding rectangle for cell
		int					n,				// number of points
		int					dim,			// dimension of space
		int& cut_dim,		// cutting dimension (returned)
		float& cut_val,		// cutting value (returned)
		int& n_lo);			// num of points on low side (returned)

	void MedianSplit(			// split points along median value
		float* pa,				// point array
		int* pidx,			// point indices
		int					n,				// number of points
		int            dd,
		int					d,				// dimension along which to split
		float& cv,			// cutting value
		int					n_lo);			// split into n_lo and n-n_lo

	AABB_node* AABB_tree(float* pa,				// point array
		int* pidx,			// point indices to store in subtree
		int					n,				// number of points
		int					dim,			// dimension of space
		int					bsp,			// bucket space
		float* bnd_box_lo,
		float* bnd_box_hi);		// bounding box for current node

public:
	float* pts;
	int* pidx;
	AABB_node* root;
	float* bnd_box_lo;			// bounding box low point
	float* bnd_box_hi;         // bounding box high point

	int dim;				// dimension of space
	int n_pts;				// number of points in tree
	int	bkt_size;			// bucket size
	int leafn;
	int nonln;
};

//----------------------------------------------------------------------
//	Leaf AABB-tree node
//		Leaf nodes of the AABB-tree store the set of points associated
//		with this bucket, stored as an array of point indices.  These
//		are indices in the array points, which resides with the
//		root of the AABB-tree.  We also store the number of points
//		that reside in this bucket and the bounding box of the node.
//----------------------------------------------------------------------

class AABB_leaf :public AABB_node                  		// leaf node for AABB-tree
{
public:
	int					n_pts;			// no. points in bucket
	int* bkt;			// bucket of points
	float* bbx; //bounding box information

public:
	AABB_leaf(							// constructor
		int			n,				// number of points
		int         dim,
		int* b,                  // bucket
		float* bndbox)
	{
		n_pts = n;			// number of points in bucket
		bkt = new int[n];
		bkt = b;			// the bucket
		bbx = new float[2 * dim];
		bbx = bndbox;
	}

	~AABB_leaf() 					// destructor (none)
	{
		if (bkt != NULL)
			delete[] bkt;
		if (bbx != NULL)
			delete[] bbx;
	}
	virtual void print() { printf("leaf node\n"); }	// print node

};

//----------------------------------------------------------------------
//	AABB-tree splitting node.
//		Splitting nodes contain the bounding box and pointers to the 2 children.
//----------------------------------------------------------------------

class AABB_split :public AABB_node                   		// split node for AABB-tree
{
public:

	float* bbx;	//bounding box
	AABB_node* child[2];					// left and right children

public:
	AABB_split(float* bndbox,
		int dim,
		AABB_node* lc,
		AABB_node* hc)		// children
	{
		bbx = new float[2 * dim];
		bbx = bndbox;
		child[0] = lc;
		child[1] = hc;
	}

	~AABB_split()						// destructor
	{
		if (bbx != NULL)
			delete[] bbx;
		if (child[0] != NULL)
			delete child[0];
		if (child[1] != NULL)
			delete child[1];
	}					// destructor (none)

	virtual void print() { printf("none-leaf node\n"); }

};



#endif