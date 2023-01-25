#include "SRStree.h"
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <float.h>
#include <cmath>

//----------------------------------------------------------------------
//	aabbSpread - find spread along given dimension
//	aabbMinMax - find min and max coordinates along given dimension
//----------------------------------------------------------------------
float SRStree::aabbSpread(				// compute point spread along dimension
	float* pa,				// point array
	int* pidx,			// point indices
	int					n,				// number of points
	int         dim,                //dimension of points
	int					d)				// dimension to check
{
	float min = pa[dim * pidx[(0)] + d];				// compute max and min coords
	float max = pa[dim * pidx[(0)] + d];
	for (int i = 1; i < n; i++) {
		float c = pa[dim * pidx[(i)] + d];
		if (c < min) min = c;
		else if (c > max) max = c;
	}
	return (max - min);					// total spread is difference
}

int SRStree::aabbMaxSpread(				// compute point spread along dimension
	float* pa,				// point array
	int* pidx,			// point indices
	int					n,				// number of points
	int					d)
{
	int max_dim = 0;					// dimension of max spread
	float max_spr = 0;				// amount of max spread

	if (n == 0) return max_dim;			// no points

	for (int i = 0; i < d; i++) {		// compute spread along each dim
		float spr = aabbSpread(pa, pidx, n, dim, i);
		if (spr > max_spr) {			// bigger than current max
			max_spr = spr;
			max_dim = i;
		}
	}
	return max_dim;
}

void SRStree::aabbMinMax(					// compute min and max coordinates along dim
	float* pa,				// point array
	int* pidx,			// point indices
	int					n,				// number of points
	int         dim,
	int					d,				// dimension to check
	float& min,			// minimum value (returned)
	float& max)			// maximum value (returned)
{
	min = pa[dim * pidx[(0)] + d];						// compute max and min coords
	max = pa[dim * pidx[(0)] + d];
	for (int i = 1; i < n; i++) {
		float c = pa[dim * pidx[(i)] + d];
		if (c < min) min = c;
		else if (c > max) max = c;
	}
}

//----------------------------------------------------------------------
//	aabbPlaneSplit - split point array about a cutting plane
//		Split the points in an array about a given plane along a
//		given cutting dimension.  On exit, br1 and br2 are set so
//		that:
//		
//				pa[ 0 ..br1-1] <  cv
//				pa[br1..br2-1] == cv
//				pa[br2.. n -1] >  cv
//
//		All indexing is done indirectly through the index array pidx.
//
//----------------------------------------------------------------------

void SRStree::aabbPlaneSplit(				// split points by a plane
	float* pa,				// points to split
	int* pidx,			// point indices
	int					n,				// number of points
	int         dim,
	int					d,				// dimension along which to split
	float			cv,				// cutting value
	int& br1,			// first break (values < cv)
	int& br2)			// second break (values == cv)
{
	int l = 0;
	int r = n - 1;
	for (;;) {							// partition pa[0..n-1] about cv
		while (l < n && pa[dim * pidx[(l)] + d] < cv) l++;
		while (r >= 0 && pa[dim * pidx[(r)] + d] >= cv) r--;
		if (l > r) break;
		int tmp = pidx[l];
		pidx[l] = pidx[r];
		pidx[r] = tmp;
		l++; r--;
	}
	br1 = l;					// now: pa[0..br1-1] < cv <= pa[br1..n-1]
	r = n - 1;
	for (;;) {							// partition pa[br1..n-1] about cv
		while (l < n && pa[dim * pidx[(l)] + d] <= cv) l++;
		while (r >= br1 && pa[dim * pidx[(r)] + d] > cv) r--;
		if (l > r) break;
		int tmp = pidx[l];
		pidx[l] = pidx[r];
		pidx[r] = tmp;
		l++; r--;
	}
	br2 = l;					// now: pa[br1..br2-1] == cv < pa[br2..n-1]
}


//----------------------------------------------------------------------
//		Find the dimension of the greatest spread, and split
//		just before the median point along this dimension.
//----------------------------------------------------------------------
void SRStree::median_split(
	float* pa,				// point array
	int* pidx,			// point indices (permuted on return)
	float* bnds,			// bounding rectangle for cell
	int					n,				// number of points
	int					dim,			// dimension of space
	int& cut_dim,		// cutting dimension (returned)
	float& cut_val,		// cutting value (returned)
	int& n_lo)
{
	cut_dim = aabbMaxSpread(pa, pidx, n, dim);
	n_lo = n / 2;							// median rank
										// split about median
	MedianSplit(pa, pidx, n, dim, cut_dim, cut_val, n_lo);
}

//----------------------------------------------------------------------
//	MedianSplit - split point array about its median
//		Splits a subarray of points pa[0..n] about an element of given
//		rank (median: n_lo = n/2) with respect to dimension d.  It places
//		the element of rank n_lo-1 correctly (because our splitting rule
//		takes the mean of these two).  On exit, the array is permuted so
//		that:
//
//		pa[0..n_lo-2][d] <= pa[n_lo-1][d] <= pa[n_lo][d] <= pa[n_lo+1..n-1][d].
//
//		The mean of pa[n_lo-1][d] and pa[n_lo][d] is returned as the
//		splitting value.
//
//		All indexing is done indirectly through the index array pidx.
//
//		This function uses the well known selection algorithm due to
//		C.A.R. Hoare.
//----------------------------------------------------------------------

void SRStree::MedianSplit(			// split points along median value
	float* pa,				// point array
	int* pidx,			// point indices
	int					n,				// number of points
	int                 dd,
	int					d,				// dimension along which to split
	float& cv,			// cutting value
	int					n_lo)			// split into n_lo and n-n_lo
{
	int l = 0;							// left end of current subarray
	int r = n - 1;						// right end of current subarray
	while (l < r) {
		int i = (r + l) / 2;		// select middle as pivot
		int k;

		if (pa[dd * pidx[(i)] + d] > pa[dd * pidx[(r)] + d])			// make sure last > pivot
		{
			int tmp = pidx[i]; pidx[i] = pidx[r]; pidx[r] = tmp;
		}
		int tmp = pidx[l]; pidx[l] = pidx[i]; pidx[i] = tmp;
		// move pivot to first position

		float c = pa[dd * pidx[(l)] + d];			// pivot value
		i = l;
		k = r;
		for (;;) {						// pivot about c
			while (pa[dd * pidx[(++i)] + d] < c);
			while (pa[dd * pidx[(--k)] + d] > c);
			if (i < k)
			{
				int tmpp = pidx[i]; pidx[i] = pidx[k]; pidx[k] = tmpp;
			}
			else break;
		}
		tmp = pidx[l]; pidx[l] = pidx[k]; pidx[k] = tmp;
		// pivot winds up in location k

		if (k > n_lo)	   r = k - 1;		// recurse on proper subarray
		else if (k < n_lo) l = k + 1;
		else break;						// got the median exactly
	}
	if (n_lo > 0) {						// search for next smaller item
		float c = pa[dd * pidx[(0)] + d];			// candidate for max
		int k = 0;						// candidate's index
		for (int i = 1; i < n_lo; i++) {
			if (pa[dd * pidx[(i)] + d] > c) {
				c = pa[dd * pidx[(i)] + d];
				k = i;
			}
		}
		int tmp = pidx[n_lo - 1]; pidx[n_lo - 1] = pidx[k]; pidx[k] = tmp;
		// max among pa[0..n_lo-1] to pa[n_lo-1]
	}
	// cut value is midpoint value
	cv = (pa[dd * pidx[(n_lo - 1)] + d] + pa[dd * pidx[(n_lo)] + d]) / 2.0;
}


AABB_node* SRStree::AABB_tree(float* pa,				// point array
	int* pidx,			// point indices to store in subtree
	int					n,				// number of points
	int					dim,			// dimension of space
	int					bsp,			// bucket space
	float* bbx_lo,
	float* bbx_hi)		// bounding box for current node
{
	float* bnd_box = new float[2 * dim];
	for (int i = 0; i < dim; i++)
	{
		bnd_box[i] = bbx_lo[i];
		bnd_box[dim + i] = bbx_hi[i];
	}

	if (n <= bsp) {
		return new AABB_leaf(n, dim, pidx, &bnd_box[0]); // construct the node and return
	}

	int cd;							// cutting dimension
	float cv;					// cutting value
	int n_lo;						// number on low side of cut

									// invoke splitting procedure
	median_split(pa, pidx, bnd_box, n, dim, cd, cv, n_lo);

	float lv = bbx_lo[cd];	// save bounds for cutting dimension
	float hv = bbx_hi[cd];

	bbx_hi[cd] = cv;			// modify bounds for left subtree
	AABB_node* lo = AABB_tree(		// build left subtree
		pa, pidx, n_lo,			// ...from pidx[0..n_lo-1]
		dim, bsp, bbx_lo, bbx_hi);
	bbx_hi[cd] = hv;			// restore bounds

	bbx_lo[cd] = cv;			// modify bounds for right subtree
	AABB_node* hi = AABB_tree(		// build right subtree
		pa, pidx + n_lo, n - n_lo,// ...from pidx[n_lo..n-1]
		dim, bsp, bbx_lo, bbx_hi);
	bbx_lo[cd] = lv;			// restore bounds
	// create the splitting node
	AABB_split* tempp = new AABB_split(&bnd_box[0], dim, lo, hi);
	return tempp;
}

//----------------------------------------------------------------------
//	Compute the smallest rectangle enclosing a set of points.
//----------------------------------------------------------------------

void SRStree::EnclRect(				// compute smallest enclosing rectangle
	float* pa,				// point array
	int* pidx,			// point indices
	int						n,				// number of points
	int						dim,			// dimension
	float* bnds_lo,
	float* bnds_hi)					// bounding cube (returned)
{
	for (int d = 0; d < dim; d++) {		// find smallest enclosing rectangle
		float lo_bnd = pa[dim * pidx[(0)] + d];		// lower bound on dimension d
		float hi_bnd = pa[dim * pidx[(0)] + d];;		// upper bound on dimension d

		for (int i = 0; i < n; i++) {
			if (pa[dim * pidx[(i)] + d] < lo_bnd) lo_bnd = pa[dim * pidx[(i)] + d];
			else if (pa[dim * pidx[(i)] + d] > hi_bnd) hi_bnd = pa[dim * pidx[(i)] + d];
		}

		bnds_lo[d] = lo_bnd;
		bnds_hi[d] = hi_bnd;
	}

}

//----------------------------------------------------------------------
// AABB_tree constructor
//----------------------------------------------------------------------
SRStree::SRStree(float* paa,							// build from point array
	int				n,				// number of points
	int				dd,				// dimension
	int				bs)
{
	pts = paa;

	dim = dd;
	n_pts = n;
	bkt_size = bs;

	if (n == 0) return;					// no points

	bnd_box_lo = new float[dim];			// bounding box for points
	bnd_box_hi = new float[dim];

	pidx = new int[n];
	for (int i = 0; i < n; i++)
	{
		pidx[i] = i;
	}
	// construct bounding rectangle
	EnclRect(paa, pidx, n, dd, bnd_box_lo, bnd_box_hi);

	leafn = 0;
	nonln = 0;
	// build by median split
	root = AABB_tree(pts, pidx, n, dd, bs, bnd_box_lo, bnd_box_hi);
}

SRStree::~SRStree()
{
	if (root != NULL) delete root;
	if (pidx != NULL) delete[] pidx;
	if (bnd_box_lo != NULL) delete[] bnd_box_lo; bnd_box_lo = NULL;
	if (bnd_box_hi != NULL) delete[] bnd_box_hi; bnd_box_hi = NULL;
}

