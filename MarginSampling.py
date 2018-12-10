#!/usr/bin/env python3


from Sampling import Sampler
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
from pdb import set_trace
import logging, sys


class MarginSampler(Sampler):
    '''Samples datapoints with the smallest margins.
    
    Based on Section 2.2 of this paper
    http://www.cs.columbia.edu/~prokofieva/CandidacyPapers/Chen_AL.pdf
    '''
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
        logging.disable(logging.DEBUG)  # Comment out to turn on logging

        # Fit logistic regression to compute posteriors of all samples
        lr = LogisticRegression()
        logging.debug("Fitting logistic regression to compute posteriors...")
        lr.fit(X_train, y_train)
        logging.debug("Finished fitting logistic regression")
        self.posteriors = lr.predict_proba(X_train)

        # Margin is measured as follows for sample n:
        # M_n = || Pr(c|x_n) - Pr(c'|x_n) ||
        # Where c is the most likely class for x_n and c' is the second most likely
        # class for x_n
        sorted_margins = np.sort(self.posteriors, axis=1) 
        self.margins = np.abs(sorted_margins[:,0] - sorted_margins[:,1])

        self.sample_indices = np.argsort(self.margins)
        self.num_sampled = 0

    def sample(self):
        '''Return index of selected training sample.'''
        sample_idx = self.sample_indices[self.num_sampled]
        self.num_sampled += 1
        return sample_idx


if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups_vectorized
    from sklearn.model_selection import train_test_split
    from pprint import pprint

    dataset = fetch_20newsgroups_vectorized(subset='train')
    X_train = dataset.data
    y_train = dataset.target

    ms = MarginSampler(X_train, y_train)
    print(ms.sample())
    print(ms.sample())
