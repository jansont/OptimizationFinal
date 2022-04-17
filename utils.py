import numpy as np
import matplotlib.pyplot as plt
from cost_functions import V_a, gradV_a, V_b, gradV_b, V_c, gradV_c


def visualize_optimization(x_hist, cost_func, cost_func_name, algorithm_name):
    '''
    Visualize progress of optimization using level contours. 
    Args:
        x_hist :: list of np array
            History of positions evaluated. 
        cost_func :: function
            Cost function
        cost_func_name :: string
            Name of cost function
        algorithm_name :: string
            Name of algorithm used
    '''
    x_hist = np.array(x_hist)
    x1_hist = x_hist[:, 0]
    x2_hist = x_hist[:, 1]

    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    x = np.meshgrid(x1,x2)
    x = np.array(x)
    Z = cost_func(x)


    fig = plt.figure(figsize = (10,7))
    contours = plt.contour(x1, x2, Z, 20)
    plt.clabel(contours, inline = True, fontsize = 10)
    plt.title(f"Evolution of {algorithm_name} optimization of cost function {cost_func_name}", fontsize=12)
    plt.plot(x1_hist, x2_hist)
    plt.plot(x1_hist, x2_hist, '*', label = "Points evaluated")
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.colorbar()
    plt.legend(loc = "upper right")
    plt.show()