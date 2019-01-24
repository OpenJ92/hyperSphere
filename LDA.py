import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
# generate a a random sample for the number of classes you desireself.

def generate_samples(norm_samples, components):
    sample_container = []
    for i in range(len(norm_samples)):
        mean = 10*np.random.random_sample(size = (components,))
        cov = np.random.random_sample(size = (components, components))
        cov = cov @ cov.T
        #cov = (1/10)*np.eye(components)
        size = (norm_samples[i],)
        _s = np.random.multivariate_normal(mean, cov, size)
        sample_container.append(_s)
    return sample_container

sample_container = generate_samples([500,600,700,800], 2)

# all functions which follow will take the sample container as the input parameter.

def mu(sample_container):
    mu_container = []
    for samples in sample_container:
        avg_sample = samples.sum(axis = 0) /  samples.shape[0]
        mu_container.append(avg_sample)
    return mu_container

def covariance(sample_container):
    cov = [[0 for i in range(len(sample_container))] for j in range(len(sample_container))]
    for i in range(len(sample_container)):
        for j in range(len(sample_container)):
            cov[i][j] = sample_container[i] @ sample_container[j].T
    return cov
