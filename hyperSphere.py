# Lecture 0 - https://www.youtube.com/watch?v=ycJEoqmQvwg&list=PLj_l4pOO0YKhJHLQVbjpPlbNyYtT8aAQz
# Lecture 1 - https://www.youtube.com/watch?v=YNIm2Op7UPg&list=PLj_l4pOO0YKhJHLQVbjpPlbNyYtT8aAQz&index=2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

## hyperSphere(theta)
##      input:
##          theta - [float, float, float, ..., float]
##
##      output:
##          np.array() | norm(np.array) ~= 1
##
## |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##
##      examples:
##          hyperSphere([np.pi]) ~= array([-1.0000000e+00,  1.2246468e-16])
##          hyperSphere([np.pi, 0]) ~= array([-1.0000000e+00,  1.2246468e-16,  0.0000000e+00])
##          hyperSphere([np.pi, 0, np.pi/2, np.pi/6]) ~= array([-5.30287619e-17,  6.49415036e-33,  0.00000000e+00,  8.66025404e-01,5.00000000e-01])
##          hyperSphere([np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/5]) ~= array([-1.75143291e-17,  2.14488671e-33,  2.86030701e-01,  4.95419707e-01,  5.72061403e-01,  5.87785252e-01])

## hyperSphereProduct_M(theta)
##        input:
##             theta - [[float], [float, float], [float, float], ..., [float, float]]
##
##        output:
##              np.array() |
##
## |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##
##      examples:
##          (make some example calls here.)

c = lambda x, theta: np.append(np.cos(theta)*x, np.sin(theta))

def hyperSphere(theta):
    _ = c(1, theta[0])
    for i in theta[1:]:
        _ = c(_, i)
    return _

def Binary_hyperSphereProduct(_A, _B):
    c = np.outer(_A, _B)
    q = np.zeros(shape = ((_A.shape[0]*_B.shape[0]) - ((_A.shape[0] - 1)*(_B.shape[0] - 1))))
    for i in range(0, _A.shape[0]):
        for j in range(0, _B.shape[0]):
            q[i+j] += c[i][j]
    return q

def hyperSphereProduct(theta):
    input_ = [hyperSphere(domain) for domain in theta]
    return reduce((lambda x, y: Binary_hyperSphereProduct(x, y)), input_)

def proveFTA(hyperSphereProduct_):
    initial_Domain_Sample = np.random.random_sample(size = len(hyperSphereProduct_) - 1)
    hyperSphere_ = hyperSphere(initial_Domain_Sample)
    #inner_product_Loss = 1 - np.absolute(hyperSphereProduct_*(1/np.linalg.norm(hyperSphereProduct_)) @ hyperSphere_)
    #a sample of random vectors belonging to the hyperSphere in desired dimension.

    # make a sphere around the sampled initial_Domain_Sample
    # resolve hyperSphere(initial_Domain_Sample + sphere_initial_Domain_Sample)
    # resolve argmax(1 - (hyperSphereProduct_ @ hyperSphere(initial_Domain_Sample + sphere_initial_Domain_Sample)))
    #           where arg == sphere_initial_Domain_Sample
    # Then change initial_Domain_Sample to the argmax resolition of above.
    # repeat process until we've reached below some threshold.

    # Currently, this function has the opperations nessesary to carry out the above process. Look to implement
    # the above process when you get back to brooklyn.

    sample_about_Domain_Sample = np.apply_along_axis(hyperSphere, 1, 2 * np.pi * np.random.random_sample(size = (1000, len(hyperSphereProduct_) - 1)))
    resolved_samples = np.apply_along_axis((lambda x: x + hyperSphere_), 1, sample_about_Domain_Sample)
    import pdb; pdb.set_trace()

    #learning_hyperSphere_ = learning_rate * (uniform_random_sample from )D

def hyperSphereTest(dimension, numTests):
    for i in range(0, numTests):
        sample = 2 * np.pi * np.random.random_sample(size = dimension)
        hS = hyperSphere(sample)
        plt.scatter(i, np.linalg.norm(hS))
        print(hS, np.linalg.norm(hS))
    plt.show()

sigma = lambda x, parameters: reduce((lambda y, z: y + z), [parameters[i]*(x**i) for i in range(0, len(parameters))])

def hyperSpherePoly(dimension, numTests):
    for i in range(0, numTests):
        sample = 2 * np.pi * np.random.random_sample(size = dimension)
        hS = hyperSphere(sample)
        plt.plot(np.linspace(-1,1,500),sigma(np.linspace(-10,10,500),hS))
    plt.show()
