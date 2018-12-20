import numpy as np
import matplotlib.pyplot as plt

# incomplete:  how to I introduce an arbitrary amount of parameters to this function
#   Note that function L has three parameters and by extension, r_func has 3 initial
#   parameters. Is there a way to specify n parameters in one argument?

L = lambda x, c, pow: (x**pow) + c
L_con = lambda x: True if np.absolute(x) < 2 else False

def r_Func(x, c, pow, func = L, con_func = L_con, count = 0):
    #import pdb; pdb.set_trace()
    func_val = func(x, c, pow)
    if (con_func(func_val)) and (count < 500):
        count += 1
        return r_Func(func_val, c, pow, func, con_func, count)
    else:
        return count

# generalized to arbitrary amount of parameters

def r_func(_, _con, _param, _con_param, count):
    func_val = _(*_param)
    if (_con(func_val, *_con_param)) and (count < 500):
        count += 1
        _param[0] = func_val
        return r_func( _, _con, _param, _con_param, count)
    else:
        return count


sample = ((4*np.random.random_sample(size = (100000,))) - 2) + ((4 * 1j * np.random.random_sample(size = (100000,))) - 2j)

# for j in np.linspace(0, 4, 400):
#     for i in sample:
#         print(r_func(L, L_con, [0,i,j], [], 0), i, j)


# for j in range(6, 400):
#     A = np.linspace(.05, 4, 200)
#     B = [r_Func(0, k, A[j]) for k in sample]
#     for i in range(0, len(sample)):
#          #print(i, B[i], sample[i])
#          if B[i] > 10:
#              if B[i] < 25:
#                  plt.scatter(sample[i].real, sample[i].imag, s = 1, c = 'blue')
#              elif B[i] < 100:
#                  plt.scatter(sample[i].real, sample[i].imag, s = 1, c = 'red')
#              else:
#                  plt.scatter(sample[i].real, sample[i].imag, s = 1, c = 'green')
#     print(j)
#     plt.axis([-2, 2, -2, 2])
#     plt.savefig(str(j) + '.jpeg')
#     plt.close()
