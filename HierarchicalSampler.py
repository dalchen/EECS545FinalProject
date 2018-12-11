#!/usr/bin/env python3


from Sampling import Sampler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
import itertools
import numpy as np
import random
import logging, sys

from pdb import set_trace

class Node:
    def __init__(self, node_id):
        '''Node object constructor.

        id -- integer representing id in clustering
        left -- Node object representing left child in clustering tree
        right -- Node object representing right child in clustering tree
        parent -- Node object representing parent in clustering tree
        weight -- floating point number representing weight of subtree
        class_to_instances -- dictionary from class --> int tallying the number of instances seen for a class
        '''
        self.id = node_id
        self.left = None
        self.right = None
        self.parent = None
        self.weight = 0
        self.is_leaf = True
        self.class_to_instances = {}



class HierarchicalSampler(Sampler):
    '''Samples datapoints based on a hierarchical method as described in the paper.
    '''
    def __init__(self, X_train, y_train, X_unlabeled, y_unlabeled, batch_size=1):
        super().__init__(X_train, y_train, X_unlabeled, y_unlabeled)
        # Hierarchical clustering portion
        self.root = None
        self.nodes = {}
        self._construct_tree()
        self._compute_weights()

        # Hierarchical sampling portion
        self.pruning = []

        

    def _construct_tree(self):
        '''Construct tree object from clusters.

        Modifies:
        self.root
        self.nodes
        '''
        # Run clustering based on merged partition of data
        print('constructing tree')
        # TODO: merge X_train and X_unlabeled
        X_merged = X_train.toarray()
        y_merged = y_train

        # Binary tree structure
        clustering = AgglomerativeClustering()
        clustering.fit(X_merged)
        ii = itertools.count(X_merged.shape[0])

        # Convert dictionary representation of tree into linked list tree structure
        def findNode(idx):
            return self.nodes[idx] if idx in self.nodes else Node(idx)

        #print(len(clustering.children_))
        for c in clustering.children_:
            left_id, right_id, node_id = c[0], c[1], next(ii)
            left, right, node = findNode(left_id), findNode(right_id), findNode(node_id)
            left.parent = node
            right.parent = node
            node.left = left
            node.right = right
            self.nodes[node_id] = node
            self.nodes[left_id] = left
            self.nodes[right_id] = right
        #print(len(self.nodes))

        # Mark root node and any leaf nodes
        self.root = next(node_id for node_id in self.nodes if self.nodes[node_id].parent is None)
        leaves = []
        for node_id, node in self.nodes.items():
            if node.left or node.right:
                node.is_leaf = False
                leaves.append(node_id)

        assert type(self.root) is int
        #print(self.root)


    def _compute_weights(self):
        '''Update weights of all nodes in the tree.

        _construct_tree must have been called beforehand.
        This function should only be called once.

        Modifies:
        self.weight for all nodes in self.nodes
        '''
        
        # Helper function to process nodes in tree bottom-up
        # TODO: could do 1-pass instead of 2-passes. Change later
        def reverse_topological_sort():
            visited = set()
            top_sorted = []
            def visit(node_id):
                visited.add(node_id)
                node = self.nodes[node_id]
                if node.left and node.left.id not in visited:
                    visit(node.left.id)
                if node.right and node.right.id not in visited:
                    visit(node.right.id)
                top_sorted.append(node_id)
            for node_id in self.nodes:
                if node_id not in visited:
                    visit(node_id)
            return top_sorted
        
        print('computing weight for all nodes in tree')

        # First set weight as number of leaf nodes that live in its subtree
        sorted_node_ids = reverse_topological_sort()
        num_leaves = 0
        for node_id in sorted_node_ids:
            node = self.nodes[node_id]
            if node.is_leaf:
                node.weight = 1
                num_leaves += 1
            else:
                node.weight = node.left.weight + node.right.weight

        '''
        for node_id, node in self.nodes.items():
            print(str(node_id) + ": " + str(node.weight))
        '''
        # Then normalize by total number of leaves in tree
        n = 1.0 * num_leaves
        for node in self.nodes.values():
            node.weight /= n
            assert node.weight < n
        return


    def _get_upward_path(self, z, v):
        '''Get list of node_ids from z to v inclusive in an upward path.

        NOTE: z must be a descendant of v.
        Used in the "Update empirical counts and probabilities" portion of the code.
        '''
        assert type(z) is int
        assert type(v) is int

        trail = [z]
        cur = z
        parent = nodes[cur].parent
        while cur != v:
            cur = parent
            parent = nodes[cur].parent
            trail.append(cur)
            if cur is None:
                raise ValueError('broken parent in node_id {}'.format(cur)) 
        return trail


        

    def _select(self):
        '''select(P) procedure in paper.

        Selects a node from the current pruning P
        '''

        def method_1():
            '''Return node_id v in Pruning
            
            (1) choose v in Pruning with probability proportional to w_v.
            This is similar to random sampling.'''
            # Normalize weights
            weights = [self.nodes[node_id].weight for node_id in self.pruning]
            v = np.random.choice(weights, size=None, replace=True, p=weights)
            print(v)
            return v


        pass




    def sample(self):
        '''Return index of selected training sample in X_unlabeled.

        In hierarchical sampling, this procedure should select the datapoint based
        on the rest of the unsampled data as well as the structure of the tree.
        '''
        pass


    def _update(self):
        '''Update. 
        '''
        pass

if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups_vectorized
    from sklearn.model_selection import train_test_split
    from pprint import pprint

    training_size = 100
    max_unlabeled_size = 500

    dataset = fetch_20newsgroups_vectorized(subset='train')
    X_train_base = dataset.data
    y_train_base = dataset.target
    X_train, y_train = X_train_base[:training_size], y_train_base[:training_size]
    X_unlabeled, y_unlabeled = X_train_base[training_size:], y_train_base[training_size:]

    hs = HierarchicalSampler(X_train, y_train, X_unlabeled, y_unlabeled)
