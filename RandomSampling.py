#!/usr/bin/env python3


from Sampling import Sampler
import numpy as np
import random


class RandomSampler(Sampler):
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        n = X_train.shape[0]
        self.sampled_indices = set(range(n))

    def sample(self):
        '''Return index of selected training sample.'''
        return self.sampled_indices.pop()

    def reveal(self, sample_id):
        '''Return label of index of training sample.'''
        return self.y[sample_id]



if __name__ == '__main__':
    '''
    We can import this file safely into other files and use RandomSampler.
    This driver in this section is just for when you run "python3 RandomSampling.py"
    '''
    from sklearn.datasets import fetch_20newsgroups
    from pprint import pprint

    newsgroups_train = fetch_20newsgroups(subset='train')
    X_train = newsgroups_train.filenames
    y_train = [newsgroups_train.target_names[idx] for idx in newsgroups_train.target]

    for num_samples in range(1,500):
        rs = RandomSampler(X_train, y_train)
        sample_ids = [rs.sample() for s in range(num_samples)]
        labels = [rs.reveal(s) for s in sample_ids]
        # TODO: figure out how to use logistic regression on sample_ids and labels
        # and plot the accuracy of the logistic regression model
