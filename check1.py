from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups_vectorized
from pprint import pprint

#!/usr/bin/env python3
# from Sampling import Sampler
import numpy as np
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

class RandomSampler(Sampler):
    def __init__(self, X_train, y_train, X_unlabeled, y_unlabeled, batch_size=1):
        import random
        super().__init__(X_train, y_train, X_unlabeled, y_unlabeled)
        self.sampled_indices = list(range(X_unlabeled.shape[0]))
        random.shuffle(self.sampled_indices)

    def sample(self):
        '''Return index of selected training sample in X_unlabeled.'''
        return self.sampled_indices.pop()

    def reveal(self, sample_id):
        '''Return label of index of training sample in y_unlabeled.'''
        '''THIS IS REDUNDANT.'''
        return self.y_unlabeled[sample_id]

#This function takes in training and test data, calculates the logistic regression function, predicts test data, and returns the error
def calculateError(x_train, y_train, x_test, y_test, lambda_value):
    clf = LogisticRegression(random_state=0, solver='lbfgs', C=1/lambda_value, multi_class='multinomial').fit(x_train, y_train)
    y_test_predict = clf.predict(x_test)
    error = 1 - accuracy_score(y_test_predict, y_test)
    return error

if __name__ == '__main__':
    '''
    We can import this file safely into other files and use RandomSampler.
    This driver in this section is just for when you run "python3 RandomSampling.py"
    '''
    training_size = 5
    max_unlabeled_size = 30

    train_dataset = fetch_20newsgroups_vectorized(subset='train')
    X_train_base = train_dataset.data
    y_train_base = train_dataset.target
    X_train, y_train = X_train_base[:training_size], y_train_base[:training_size]
    X_unlabeled, y_unlabeled = X_train_base[training_size:], y_train_base[training_size:]

    test_dataset = fetch_20newsgroups_vectorized(subset='test')
    X_test = test_dataset.data
    y_test = test_dataset.target

    X_train_zeros = np.zeros((training_size + max_unlabeled_size, X_train.shape[1]))
    for num_samples in range(max_unlabeled_size):
        rs = RandomSampler(X_train, y_train, X_unlabeled, y_unlabeled)
        index = rs.sample()
        y_train = np.append(y_train, y_unlabeled[index])
        for j in range(X_unlabeled.shape[1]):
            X_train_zeros[num_samples+training_size,j] = X_unlabeled[index,j]
#     print(X_train_zeros)
    for num_samples in range(training_size):
        for j in range(X_train.shape[1]):
            X_train_zeros[num_samples,j] = X_train[num_samples,j]

    #Initialize parameters and total number of labeled points
    lambda_value = 10**(-4)#This needs to be tuned

    #Initialize vectors to be used for plotting
    error_random_vector = []
    num_samples_vector = []

    #Iterating through number of samples, and adding the resulting errors to plotting vectors

    i = training_size
    for num_samples in range(training_size, training_size + max_unlabeled_size):
        #Each iteration you are using more labeled data points to train
        num_samples_vector.append(num_samples)
        error_random_vector.append(calculateError(X_train_zeros[:num_samples,:],
                                                  y_train[:num_samples],
                                                  X_test,
                                                  y_test, lambda_value))
        print(i)
        i = i + 1
    #Plotting
    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(num_samples_vector, error_random_vector)
    plt.legend(['Random', 'Margin'], loc='upper right')
    plt.xlabel("Number Of Labels")
    plt.ylabel("Error")
    plt.show()
