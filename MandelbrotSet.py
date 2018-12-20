import numpy as np
import matplotlib.pyplot as plt

# incomplete:  Look to reform this itterated function so that we can generate the
#   Mandelbrot Set

L = lambda x, c: x*x
L_con = lambda x: True if np.absolute(x) < 2 else False

def r_Func(val, func = L, con_func = L_con, count = 0):
    func_val = func(val)
    #import pdb; pdb.set_trace()
    if (con_func(func_val)) and (count < 100):
        count += 1
        print(count, func_val, np.absolute(func_val))
        return r_Func(func_val, func, con_func, count)
    else:
        return count

sample = (np.random.random_sample(size = (10000,))) + (1j * np.random.random_sample(size = (10000,)))

A = [r_Func(i) for i in sample]
for i in range(0, len(sample)):
     print(i, A[i], sample[i])
     if A[i] > 50:
         plt.scatter(sample[i].real, sample[i].imag)
plt.show()
