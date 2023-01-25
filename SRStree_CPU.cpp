#include "SRStree_CPU.h"
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <float.h>
#include <cmath>
#include <typeinfo>


SRStree_CPU::SRStree_CPU()
{
	m_id = 0;
	max_level = 0;
	m_num_leafnodes = 0;
}

SRStree_CPU::~SRStree_CPU()
{
	// Delete all the ndoes
	vector <AABBNode_CPU*> to_visit;

	to_visit.push_back(m_root);

	while (to_visit.size()) {
		vector <AABBNode_CPU*> next_search;

		while (to_visit.size()) {
			AABBNode_CPU* cur = to_visit.back();
			to_visit.pop_back();

			if (cur->left)
				next_search.push_back(cur->left);

			if (cur->right)
				next_search.push_back(cur->right);

			delete cur;
		}

		to_visit = next_search;
	}

	m_root = NULL;
}

//The tree is traversed in breath-first order. During traversal, the nodes are 
//numbered. The information of their bounding box, associated bucket are recorded. 

void SRStree_CPU::Reconstruct(SRStree* aabbtree)
{
	dim = aabbtree->dim;
	n_pts = aabbtree->n_pts;
	bkt_size = aabbtree->bkt_size;

	m_root = new AABBNode_CPU();
	m_root->id = m_id++;
	m_root->level = 0;

	vector <AABB_node*> to_visit;
	vector<AABBNode_CPU*> visit;
	to_visit.push_back(aabbtree->root);
	visit.push_back(m_root);

	//breath-first traversal of the tree
	int lev = -1;
	while (to_visit.size()) {
		vector <AABB_node*> next_search;
		vector<AABBNode_CPU*> search;
		lev++;

		while (to_visit.size()) {
			AABB_node* node = to_visit.back();
			AABBNode_CPU* cur = visit.back();
			to_visit.pop_back();
			visit.pop_back();


			if (!strcmp(typeid(*node).name(), "class AABB_split")) {//for split node

				AABB_split* snode = (AABB_split*)node;

				AABBNode_CPU* left = new AABBNode_CPU();
				AABBNode_CPU* right = new AABBNode_CPU();

				left->id = m_id++;
				right->id = m_id++;

				left->level = cur->level + 1;
				right->level = cur->level + 1;

				left->parent = cur;
				right->parent = cur;

				left->_parent = cur->id;
				right->_parent = cur->id;


				vector <int> dummy;
				cur->bkt.swap(dummy);

				cur->type = 0;

				for (int i = 0; i < 2 * aabbtree->dim; i++)//bounding box
				{
					cur->bnd_box.push_back(snode->bbx[i]);
				}

				cur->left = left;
				cur->right = right;

				cur->_left = left->id;
				cur->_right = right->id;

				next_search.push_back(snode->child[0]);
				next_search.push_back(snode->child[1]);
				search.push_back(left);
				search.push_back(right);
			}
			if (!strcmp(typeid(*node).name(), "class AABB_leaf")) {//for leaf node
				AABB_leaf* snode = (AABB_leaf*)node;
				cur->type = 1;
				cur->n_pts = snode->n_pts;

				for (int i = 0; i < snode->n_pts; i++) //associated points in the bucket
				{
					cur->bkt.push_back(snode->bkt[i]);
				}

				for (int i = 0; i < 2 * aabbtree->dim; i++)//bounding box
				{
					cur->bnd_box.push_back(snode->bbx[i]);
				}

				m_num_leafnodes++;
			}
		}

		to_visit = next_search;
		visit = search;
	}

	max_level = lev;

	printf("--------------------------------------------------------------\n");
	printf("Levels of the tree: %d\n", max_level);
	printf("Number of leaf-nodes on the tree: %d\n", m_num_leafnodes);
	printf("--------------------------------------------------------------\n");
}
