import numpy as np

L = lambda x: x**2
L_con = lambda x: True if np.absolute(x) < 2 else False

def r_Func(val, func = L, con_func = L_con, count = 0):
    func_val = func(val)
    import pdb; pdb.set_trace()
    if (con_func(func_val)) and (count <= 100):
        count += 1
        print(count, func_val, np.absolute(func_val))
        return r_Func(func_val, func, con_func, count)
    else:
        return count

sample = np.random.random_sample(size = (1000,)) + 1j * np.random.random_sample(size = (1000,))
_resolved_samples = np.apply_along_axis(r_Func, 0, sample)
