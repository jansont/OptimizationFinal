import numpy as np
from algorithms import * #steepest_descent, conjugate_gradient, secant
from cost_functions import * #V_a, gradV_a, V_b, gradV_b

# class Problem():
#     def __init__(self, cost_function, gradient_function=None, ecp=None, icp=None):
#         self.V = cost_function
#         self.grad = gradient_function if gradient_function is not None else self.gradient_approx
#         self.ecp = ecp
#         self.icp = icp
    
#     def gradient_approx(x0, cost_function, h):
#         gradient = np.zeros_like(x0)
#         perturbation = np.eye(len(x0))*h
#         for i in range(len(x0)):
#             gradient[i] = (cost_function(x0+perturbation[i])-cost_function(x0))/h
#         return gradient

if __name__ == '__main__':
    
    #____________PROBLEM A________________________

    print('\n---- TESTING PROBELM A ----\n')
    
    x0 = np.zeros((6,1))

    # test steepest descent
    x, minimum =  steepest_descent(x0,
                                V_a,
                                gradV_a,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Steepest descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test conjugate gradient descent
    x, minimum =  conjugate_gradient(x0,
                                V_a,
                                gradV_a,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Conjugate descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test secant function
    x, minimum =  secant(x0,
                        V_a,
                        gradV_a,
                        step_size = 1e-4,
                        threshold = 1e-6, 
                        log = False, 
                        h = 1e-8, 
                        max_iter = 1e7, 
                        fd_method = 'central', 
                        track_history = False)
    print(f'Secant Method:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________PROBLEM B________________________

    print('\n---- TESTING PROBELM B ----\n')
    x0 = np.ones((2,1))

    # test steepest descent
    x, minimum =  steepest_descent(x0,
                                V_b,
                                gradV_b,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Steepest descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test conjugate gradient descent
    x, minimum =  conjugate_gradient(x0,
                                V_b,
                                gradV_b,
                                step_size = 1e-4,
                                threshold = 1e-8, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Conjugate descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test secant function
    x, minimum =  secant(x0,
                        V_b,
                        gradV_b,
                        step_size = 1e-4,
                        threshold = 1e-4, 
                        log = False, 
                        h = 1e-8, 
                        max_iter = 1e7, 
                        fd_method = 'central', 
                        track_history = False)
    print(f'Secant Method:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________PROBLEM C________________________

    print('\n---- TESTING PROBELM C ----\n')
    x0 = np.zeros((2,1))

    # test steepest descent
    x, minimum =  steepest_descent(x0,
                                V_c,
                                gradV_c,
                                step_size = 1e-4,
                                threshold = 1e-6, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Steepest descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test conjugate gradient descent
    x, minimum =  conjugate_gradient(x0,
                                V_c,
                                gradV_c,
                                step_size = 1e-4,
                                threshold = 1e-6, 
                                log = False, 
                                h = 1e-8, 
                                max_iter = 1e12, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Conjugate descent:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    # test secant function
    x, minimum =  secant(x0,
                        V_c,
                        gradV_c,
                        step_size = 1e-4,
                        threshold = 1e-4, 
                        log = False, 
                        h = 1e-8, 
                        max_iter = 1e7, 
                        fd_method = 'central', 
                        track_history = False)
    print(f'Secant Method:\n x={x.flatten()}, V(x)={minimum:.5f}\n')
