import numpy as np
import matplotlib.pyplot as plt

# https://newonlinecourses.science.psu.edu/stat504/node/209/

def factorial(n):
     if n > 1:
         return n * factorial(n - 1)
     else:
         return 1

def permutation(n, r):
     return factorial(n) / factorial(n-r)

def combination(n, r):
     return permutation(n, r) / factorial(r)

def bernoulli_distribution(n, p):
    return [(i, ((p)**(i))*((1-p)**(1-i))) for i in range(0, n+1)]

# def binomial_distribution(n, p):
#     A = bernoulli_distribution(n, p)
#     return [(i, A[i][1] * combination(n, i)) for i in range(0, n+1)]
#
# def hypergeometric_distribution():
#     pass
#
# def poisson_distribution():
#     pass
