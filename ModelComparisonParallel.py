#!/usr/bin/env python3

# Samplers
from Sampling import Sampler
from RandomSampling import RandomSampler
from MarginSampling import MarginSampler
from HierarchicalSampler import HierarchicalSampler
import numpy as np
# Dataset
from sklearn.datasets import fetch_20newsgroups_vectorized
# Model function
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix, vstack
# Plotting
from matplotlib import pyplot as plt
from cycler import cycler
# Multiprocessing
from multiprocessing import Process, Queue

# Dataset
dataset = fetch_20newsgroups_vectorized(subset='train')

# Parameters: Tune these!
training_size = 30
max_unlabeled_size = 500
batch_size = 10
max_samples = 1000 # new parameter: reduce size of training base before split

# Training
X_train_base = dataset.data[:max_samples]
y_train_base = dataset.target[:max_samples]
X_train, y_train = X_train_base[:training_size], y_train_base[:training_size]
X_unlabeled, y_unlabeled = X_train_base[training_size:], y_train_base[training_size:]

# Testing
testset = fetch_20newsgroups_vectorized(subset='test')
X_test = testset.data[:1000]
y_test = testset.target[:1000]

# For multiprocessing
output = Queue()

# Launcher function
def run_test(sampler_type, X_train, y_train, X_test, y_test):
    # Samplers
    sampler = None
    if sampler_type == 'rs':
        sampler = RandomSampler(X_train, y_train, X_unlabeled, y_unlabeled)
    elif sampler_type == 'ms':
        sampler = MarginSampler(X_train, y_train, X_unlabeled, y_unlabeled)
    elif sampler_type == 'hs':
        sampler = HierarchicalSampler(X_train, y_train, X_unlabeled, y_unlabeled)
    else:
        raise ValueError

    print("Finish constructing sampler class "+sampler_type)

    errors = []
    X_train, y_train = sampler.X_train, sampler.y_train
    i = 0
    while i < max_unlabeled_size:
        x_samples, y_samples = np.empty((batch_size, X_train.shape[1]),float), np.empty((batch_size,),int)
        for b in range(batch_size):
            x_sample, y_sample = sampler.sample()
            x_samples[b] = x_sample.toarray()
            y_samples[b] = y_sample
        X_train = vstack([X_train, x_samples])
        y_train = np.append(y_train, y_samples)
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
        model.fit(X_train, y_train)
        #y_pred = model.predict(X_test)
        error = 1 - model.score(X_test, y_test)
        print(sampler_type+' number of labels: '+str(training_size+i)+ ' error='+str(error))
        errors.append(error)
        i += batch_size
    output.put((sampler_type, errors))

processes = [
    Process(target=run_test, args=('rs',X_train,y_train,X_test,y_test)),
    Process(target=run_test, args=('ms',X_train,y_train,X_test,y_test)),
    Process(target=run_test, args=('hs',X_train,y_train,X_test,y_test))
]

print("Launching a process for each sampler...")

for process in processes:
    process.start()
for process in processes:
    process.join()


# Collect data from multiprocessing
errors = {}
for _ in range(len(processes)):
    sampler_type, error = output.get()
    errors[sampler_type] = error

# Plots
num_samples = list(range(len(errors['rs'])))
cy = cycler('color', ['red', 'green', 'blue'])
plt.rc('axes', prop_cycle=cy)

for sampler_type in errors:
    plt.plot(num_samples, errors[sampler_type])

plt.legend(['Random', 'Margin', 'Hierarchical'], loc='upper right')
plt.xlabel('Number of Labels')
plt.ylabel('Error')
#plt.show()
plt.savefig('20newsgroups_parallel_'+str(max_samples)+'samples.png')

print('Done')
