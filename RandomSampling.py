#!/usr/bin/env python3


from Sampling import Sampler
import numpy as np


class RandomSampler(Sampler):
    def __init__(self, X_train, y_train, X_unlabeled, y_unlabeled, batch_size=1):
        import random
        super().__init__(X_train, y_train, X_unlabeled, y_unlabeled)
        self.sampled_indices = list(range(X_unlabeled.shape[0]))
        random.shuffle(self.sampled_indices)

    def sample(self):
        '''Return selected training sample in X_unlabeled.'''
        return self.X_unlabeled[self.sampled_indices.pop()]



if __name__ == '__main__':
    '''
    We can import this file safely into other files and use RandomSampler.
    This driver in this section is just for when you run "python3 RandomSampling.py"
    '''
    from sklearn.datasets import fetch_20newsgroups_vectorized
    from pprint import pprint

    training_size = 100
    max_unlabeled_size = 500

    dataset = fetch_20newsgroups_vectorized(subset='train')
    X_train_base = dataset.data
    y_train_base = dataset.target
    X_train, y_train = X_train_base[:training_size], y_train_base[:training_size]
    X_unlabeled, y_unlabeled = X_train_base[training_size:], y_train_base[training_size:]

    for num_samples in range(1,max_unlabeled_size):
        rs = RandomSampler(X_train, y_train, X_unlabeled, y_unlabeled)
        print(rs.sample())
        print(rs.sample())
        print(rs.sample())
        # TODO: figure out how to use logistic regression on sample_ids and labels
        # and plot the accuracy of the logistic regression model
