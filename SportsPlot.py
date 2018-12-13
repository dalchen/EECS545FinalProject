#!/usr/bin/env python3

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups_vectorized
from pprint import pprint
import json
import gzip
import pandas as pd
import gzip
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, vstack
import csv
import numpy as np

from Sampling import Sampler
from RandomSampling import RandomSampler
from MarginSampling import MarginSampler
from HierarchicalSampler import HierarchicalSampler

#This function takes in training and test data, calculates the logistic regression function, predicts test data, and returns the error
def calculateError(x_train, y_train, x_test, y_test, lambda_value):
    clf = LogisticRegression(random_state=0, solver='lbfgs', C=1/lambda_value, multi_class='multinomial').fit(x_train, y_train)
    # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train)
    y_test_predict = clf.predict(x_test)
    error = 1 - accuracy_score(y_test_predict, y_test)
    return error

def Plotting(training_size, max_unlabeled_size, x_test, y_test, x_train_random, y_train_random, x_train_margin, y_train_margin, x_train_Hierarchical, y_train_Hierarchical, lambda_value):


    # Initialize vectors to be used for plotting
    error_random_vector = []
    error_margin_vector = []
    error_Hierarchical_vector = []
    num_samples_vector = []

    # Iterating through number of samples, and adding the resulting errors to plotting vectors
    i = training_size + 1
    for num_samples in range(training_size, training_size + max_unlabeled_size):
        # Each iteration you are using more labeled data points to train
        if(i%10 ==0):
            num_samples_vector.append(num_samples + 1)
            error_random_vector.append(calculateError(x_train_random[:num_samples, :], y_train_random[:num_samples], x_test, y_test, lambda_value))
            error_margin_vector.append(calculateError(x_train_margin[:num_samples, :], y_train_margin[:num_samples], x_test, y_test, lambda_value))
            error_Hierarchical_vector.append(calculateError(x_train_Hierarchical[:num_samples, :], y_train_Hierarchical[:num_samples], x_test, y_test, lambda_value))
            print(i)
        i = i + 1

    # Plotting
    data_set_title = "Sports"
    plt.gca().set_color_cycle(['red', 'green', 'blue'])
    plt.plot(num_samples_vector, error_random_vector)
    plt.plot(num_samples_vector, error_margin_vector)
    plt.plot(num_samples_vector, error_Hierarchical_vector)
    plt.legend(['Random', 'Margin', 'Hierarchical'], loc='upper right')
    plt.xlabel("Number Of Labels")
    plt.ylabel("Error")
    # plt.show()
    plt.savefig(data_set_title+str("_")+str(lambda_value)+".jpg")
    print("saved " + data_set_title+str("_")+str(lambda_value)+".jpg")
    plt.clf()


if __name__ == '__main__':
    '''
    We can import this file safely into other files and use RandomSampler.
    This driver in this section is just for when you run "python3 RandomSampling.py"
    '''

    print("Start")
    training_size = 5#100
    max_unlabeled_size = 750#400
    lambda_value = 0#10**(-4)#This needs to be tuned
    num_training_data = 800

    #Sports Data
    data = pd.read_csv('sportsData.csv')
    data = data.sample(frac=1)
    data = data.values
    x_data = data[:,:-1]
    y_data = data[:,-1]
    X_train_base = x_data[:num_training_data,:]
    y_train_base = y_data[:num_training_data]
    X_test = x_data[num_training_data:,:]
    y_test = y_data[num_training_data:]
    X_train, y_train = X_train_base[:training_size], y_train_base[:training_size]
    X_unlabeled, y_unlabeled = X_train_base[training_size:], y_train_base[training_size:]

    rs = RandomSampler(X_train, y_train, X_unlabeled, y_unlabeled)
    ms = MarginSampler(X_train, y_train, X_unlabeled, y_unlabeled)
    hs = HierarchicalSampler(X_train, y_train, X_unlabeled, y_unlabeled)

    x_train_random = X_train
    y_train_random = y_train
    x_train_margin = X_train
    y_train_margin = y_train
    x_train_Hierarchical = X_train
    y_train_Hierarchical = y_train

    print('Successfully loaded the Sports dataset into train and test set.')

    for num_samples in range(max_unlabeled_size):
        #Add data, random
        x_sample, y_sample = rs.sample()
        x_sample = np.array([x_sample])
        x_train_random = vstack([x_train_random, x_sample]).toarray()
        y_train_random = np.append(y_train_random, y_sample)
        #Add data, margin
        x_sample, y_sample = ms.sample()
        x_sample = np.array([x_sample])
        x_train_margin = vstack([x_train_margin, x_sample]).toarray()
        y_train_margin = np.append(y_train_margin, y_sample)
        #Add data, Hierarchical
        x_sample, y_sample = hs.sample()
        x_sample = np.array([x_sample])
        x_train_Hierarchical = vstack([x_train_Hierarchical, x_sample]).toarray()
        y_train_Hierarchical = np.append(y_train_Hierarchical, y_sample)

    for i in range(-8,9):
        lambda_value = 10**(i)
        print("Lambda = " + str(lambda_value))
        Plotting(training_size, max_unlabeled_size, X_test, y_test, x_train_random, y_train_random, x_train_margin, y_train_margin, x_train_Hierarchical, y_train_Hierarchical, lambda_value)
