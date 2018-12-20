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

def L(Domain_):
    if len(Domain_) == 2:
        return np.array([Domain_[0]], [Domain_[1]])
    hyperSphereProductDomain_ = [[Domain_[element], Domain_[element + 1]]for element in 2 * np.array(range(0, int(len(Domain_) / 2)))]
    if len(Domain_) % 2 == 1:
        hyperSphereProductDomain_.insert(0, [Domain_[-1]])
    return hyperSphereProductDomain_


def q(hyperSphereProductDomain_, num_samples = 10000, learning_rate = .01):
    # import pdb; pdb.set_trace()
    # forming the above hyperSphereProductDomain_ is not easy given in the input
    # above, The function L(Domain_) works to fix this so that one can generate a
    # matrix np.random.random_sample(size = (1.000.000, dimension)) and apply along the
    # first axis our q function. ie:
    #           np.apply_along_axis(q, 1, np.random.random_sample(size = (1.000.000, dimension)), num_samples, learning_rate)
    #              **Look into how one applies additional arguments in np.apply_along_axis
    hyperSphereProduct_ = hyperSphereProduct(hyperSphereProductDomain_)
    hyperSphereProduct_ = (1 / np.linalg.norm(hyperSphereProduct_)) * hyperSphereProduct_

    domain_Sample = 2 * np.pi * np.random.random_sample(size = (num_samples, len(hyperSphereProduct_) - 1))
    hyperSphere_Sample = np.apply_along_axis(hyperSphere, 1, domain_Sample)

    innerProduct = hyperSphere_Sample @ hyperSphereProduct_
    argmaxinnerProduct = domain_Sample[np.argmax(innerProduct)]

    w = 1
    while(0 < 1):
        subDomain_sample = np.random.random_sample(size = (5000, len(hyperSphereProduct_) - 2))
        domain_hyperSphere_Sample_local = learning_rate * np.apply_along_axis(hyperSphere, 1, subDomain_sample)
        domain_hyperSphere_Sample_local = np.apply_along_axis((lambda x: np.random.random_sample() * x), 1, domain_hyperSphere_Sample_local)
        domain_hyperSphere_Sample_global = np.apply_along_axis((lambda x: x + argmaxinnerProduct), 1, domain_hyperSphere_Sample_local)
        A = np.apply_along_axis(hyperSphere, 1, domain_hyperSphere_Sample_global)
        B = A @ hyperSphereProduct_
        if (hyperSphere(argmaxinnerProduct) @ hyperSphereProduct_) < B[np.argmax(B)]:
            argmaxinnerProduct = domain_hyperSphere_Sample_global[np.argmax(B)]
            w = 0
        else:
            w += 1
            if w >= 10:
                return hyperSphereProduct_, hyperSphereProductDomain_, argmaxinnerProduct, hyperSphere(argmaxinnerProduct), hyperSphere(argmaxinnerProduct) @ hyperSphereProduct_,learning_rate

def testq(num_, dimension_):
    samp = np.random.random_sample(size = (num_, dimension_))
    samp_A = np.apply_along_axis(L, 1, samp)
    samp_B = np.apply_along_axis(q, 1, samp_A)
    return samp_B

# q([np.pi*2*np.random.random_sample(size = (1,)), np.pi*2*np.random.random_sample(size = (2,)), np.pi*2*np.random.random_sample(size = (1,))], 10000, .05)

def p(Domain_):
    #if len(Domain_) % 2 == 0:
    hyperSphereProductDomain_ = [[Domain_[element], Domain_[element + 1]]for element in 2 * np.array(range(0, int(len(Domain_) / 2)))]
    if len(Domain_) % 2 == 1:
        hyperSphereProductDomain_.insert(0, [Domain_[-1]])
    import pdb; pdb.set_trace()

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
