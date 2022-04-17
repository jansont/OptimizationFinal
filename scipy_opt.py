import numpy as np
from scipy.optimize import minimize
from cost_functions import *

x0 = np.zeros((6,1))
res_a = minimize(V_a, x0)
minimum = V_a(res_a.x).item()
print(f'Unconstrained A:\n x={res_a.x}, V(x)={minimum:.5f}\n')

x0 = np.ones((2,1))
res_b = minimize(V_b, x0)
minimum = V_b(res_b.x).item()
print(f'Unconstrained B:\n x={res_b.x}, V(x)={minimum:.5f}\n')

x0 = np.ones((2,1))
res_c = minimize(V_c, x0)
minimum = V_c(res_c.x).item()
print(f'Unconstrained C:\n x={res_c.x}, V(x)={minimum:.5f}\n')
