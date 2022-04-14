import numpy as np
from algorithms import steepest_descent, conjugate_gradient, secant
from cost_functions import V_a, gradV_a, V_b, gradV_b

x0 = np.zeros((6,))
x, minimum =  steepest_descent(x0,
                               V_a,
                               gradV_a,
                               step_size = 1e-4,
                               threshold = 1e-8, 
                               log = True, 
                               h = 1e-8, 
                               max_iter = 1e12, 
                               fd_method = 'central', 
                               track_history = False)
