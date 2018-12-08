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
##          hyperSphere([np.pi]) ~= [1,0]
##          hyperSphere([np.pi, 0]) ~= [1,0,0]
##          hyperSphere([0,0,0,0, ...,0]) ~= [1,0,0,0, ...,0]
##          hyperSphere([0,0,0,0, ...,0]) ~= [0,0,0,0, ...,1]

def hyperSphere(theta):
    _ = c(1, theta[0])
    for i in theta[1:]:
        _ = c(_, i)
    return _

def hyperSphereTest(dimension, numTests):
    for i in range(0, numTests):
        sample = np.pi * np.random.random_sample(size = dimension)
        hS = hyperSphere(sample)
        plt.scatter(i, np.linalg.norm(hyperSphere(sample)))
    plt.show()

def hyperSpherePoly(dimension, numTests):
    for i in range(0, numTests):
        sample = 2*np.pi*np.random.random_sample(size = dimension)
        hS = hyperSphere(sample)
        plt.plot(np.linspace(-1,1,500),sigma(np.linspace(-10,10,500),hS))
    plt.show()
