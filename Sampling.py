#!/usr/bin/env python3

'''
How to use: add "from Sampling import Sampler" into your inherited sampling class.
'''

class Sampler:
    '''
    This is the base class for which the 3 different sampling classes
    (Random, Uncertainty, and Hierarchical) inherit from.
    '''
    def __init__(self, X_train, y_train, X_unlabeled, batch_size=1):
        self.X_train = X_train
        self.y_train = y_train
        self.X_unlabeled = X_unlabeled
        self.batch_size = batch_size

    def sample(self):
        pass

    def reveal(self, sample_id):
        pass


