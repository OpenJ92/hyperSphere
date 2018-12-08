# Lecture 0 - https://www.youtube.com/watch?v=ycJEoqmQvwg&list=PLj_l4pOO0YKhJHLQVbjpPlbNyYtT8aAQz
# Lecture 1 - https://www.youtube.com/watch?v=YNIm2Op7UPg&list=PLj_l4pOO0YKhJHLQVbjpPlbNyYtT8aAQz&index=2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

sigma = lambda x, parameters: reduce((lambda y, z: y + z), [parameters[i]*(x**i) for i in range(0, len(parameters))])
c = lambda x, theta: np.append(np.cos(theta)*x, np.sin(theta))

## hyperSphere(theta)
##      input:
##          theta - (itterable) (float) of arbitrary length
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

def hyperSphere(theta):
    _ = c(1, theta[0])
    for i in theta[1:]:
        _ = c(_, i)
    return _

def hyperSphereTest(dimension, numTests):
    for i in range(0, numTests):
        sample = 2 * np.pi * np.random.random_sample(size = dimension)
        hS = hyperSphere(sample)
        plt.scatter(i, np.linalg.norm(hyperSphere(sample)))
    plt.show()

def hyperSpherePoly(dimension, numTests):
    for i in range(0, numTests):
        sample = 2 * np.pi * np.random.random_sample(size = dimension)
        hS = hyperSphere(sample)
        plt.plot(np.linspace(-1,1,500),sigma(np.linspace(-10,10,500),hS))
    plt.show()
