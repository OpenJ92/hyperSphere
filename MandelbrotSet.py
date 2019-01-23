import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go

L = lambda x, c, pow: (x**pow) + c
L_con = lambda x, count: True if (np.absolute(x)) < 2 and (count < 500) else False

m = lambda x, mu, sigma: np.e**(-1*(x - mu)**2 / 2*np.sqrt(sigma))
m_con = lambda x, count: True if count <= 500 else False

# r_func usage:
# r_func takes in a defined function (_) and a condition function (_con). Notice that
# in this case, _(*_param) must evaluate to an element resolvable by _. In other words,
# it must be _(*_param) =~ *_param. Secondly,
def r_func(_, _con, _param, _con_param):
    func_val = _(*_param)
    if _con(func_val, *_con_param):
        print(_, _con, _param, _con_param)
        _con_param[-1] += 1
        _param[0] = func_val
        return r_func(_, _con, _param, _con_param)
    else:
        return (_, _con, _param, _con_param)

def plot_complex(complex):
    return plt.scatter(complex.real, complex.imag, c = 'blue')

sample = ((4*np.random.random_sample(size = (75000,))) - 2) + ((4 * 1j * np.random.random_sample(size = (75000,))) - 2j)

# r = []
# for j in np.linspace(0, 4, 400):
#     q = []
#     for i in sample:
#         B = r_func(L, L_con, [0,i,j], [0])
#         q.append(B)
#     r.append(q)

# A = np.array(r)

# for j in range(100, 200):
#     A = np.linspace(.05, 4, 200)
#     #A = np.linspace(4, 8, 200) # Next
#     B = [r_Func(0, k, A[j]) for k in sample]
#     for i in range(0, len(sample)):
#          if B[i] > 10:
#              print(i, B[i], sample[i])
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


# def r_Func(x, c, pow, func = L, con_func = L_con, count = 0):
#     func_val = func(x, c, pow)
#     if (con_func(func_val)) and (count < 500):
#         count += 1
#         return r_Func(func_val, c, pow, func, con_func, count)
#     else:
#         return count
