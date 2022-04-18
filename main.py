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

    print('\n---- TESTING PROBLEM A ----\n')
    
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

    print('\n---- TESTING PROBLEM B ----\n')
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

    print('\n---- TESTING PROBLEM C ----\n')
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

    #____________CONSTRAINED PROBLEM 1________________________

    print('\n---- TESTING CONSTRAINED PROBLEM 1 ----\n')
    x0 = np.array([[0.1],[0.7]])

    # test penalty function
    x, minimum =  penalty_fn(x0,
                                V_1,
                                gradient_function=None,
                                ecp=h2_1,
                                icp=[h1_1],
                                step_size = 1e-4,
                                threshold = 1e-3, 
                                conv_threshold=1e-3,
                                log = False, 
                                h = 1e-5, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Penalty function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.6],[0.6]])

    # test barrier function
    x, minimum =  barrier_fn(x0,
                                V_1,
                                gradient_function=None,
                                mode='inv',
                                ecp=h2_1,
                                icp=[h1_1],
                                step_size = 1e-4,
                                threshold = 1e-3,
                                conv_threshold = 1e-6,  
                                log = False, 
                                h = 1e-5, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Barrier function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________CONSTRAINED PROBLEM 2________________________

    print('\n---- TESTING CONSTRAINED PROBLEM 2 ----\n')
    x0 = np.array([[1.],[-1.]])

    # test penalty function
    x, minimum =  penalty_fn(x0,
                                V_2,
                                gradient_function=None,
                                ecp=None,
                                icp=[h1_2,h2_2],
                                step_size = 1e-4,
                                threshold = 1e-3,
                                conv_threshold=1e-3,
                                log = False, 
                                h = 1e-7, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Penalty function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    x0 = np.array([[0.5],[0.5]])

    # test barrier function
    x, minimum =  barrier_fn(x0,
                                V_2,
                                gradient_function=None,
                                mode='inv',
                                ecp=None,
                                icp=[h1_2,h2_2],
                                step_size = 1e-4,
                                threshold = 1e-7,
                                conv_threshold = 1e-6, 
                                log = False, 
                                h = 1e-5, 
                                max_iter = 1e5, 
                                fd_method = 'central', 
                                track_history = False)
    print(f'Barrier function:\n x={x.flatten()}, V(x)={minimum:.5f}\n')

    #____________CONSTRAINED PROBLEM 3________________________

    print('\n---- TESTING CONSTRAINED PROBLEM 3 ----\n')
    x0 = np.array([[4.],[2.]])

    #test penalty function
    x, _ =  penalty_fn(x0,
                        V_3,
                        gradient_function=None,
                        ecp=h2_3,
                        icp=[h1_3],
                        sigma_max=1e3,
                        step_size = 1e-4,
                        threshold = 1e-3,
                        conv_threshold=1e-3, 
                        log = False, 
                        h = 1e-7, 
                        max_iter = 1e5, 
                        fd_method = 'forward', 
                        track_history = False)
    V = np.log(x[0]) - x[1]
    print(f'Penalty function:\n x={x.flatten()}, V(x)={V.item():.5f}\n')

    x0 = np.array([[2.],[2.]])

    # test barrier function
    x, _ =  barrier_fn(x0,
                        V_3,
                        gradient_function=None,
                        ecp=h2_3,
                        icp=[h1_3],
                        step_size = 1e-4,
                        threshold = 1e-3,
                        conv_threshold=1e-6,
                        log = False, 
                        h = 1e-7, 
                        max_iter = 1e5, 
                        fd_method = 'forward', 
                        track_history = False)
    V = np.log(x[0]) - x[1]
    print(f'Barrier function:\n x={x.flatten()}, V(x)={V.item():.5f}\n')