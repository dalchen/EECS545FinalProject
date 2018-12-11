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
        self.weight = None
        self.class_to_instances = {}


class HierarchicalSampler(Sampler):
    '''Samples datapoints based on a hierarchical method as described in the paper.
    '''
    def __init__(self, X_train, y_train, X_unlabeled, y_unlabeled, batch_size=1):
        # Hierarchical clustering portion
        self.root = None
        self.nodes = {}
        self._construct_tree()
        

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

        self.root = next(node_id for node_id in self.nodes if self.nodes[node_id].parent is None)
        assert type(self.root) is int
        print(self.root)


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
