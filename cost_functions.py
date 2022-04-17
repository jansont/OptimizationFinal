import numpy as np
from numpy.linalg import norm, eig

#-_____________COST FUNCTION A___________________
#COST FUNCTION A
def V_a(x):
    if x.shape[0] != 6:
        raise ValueError('X must have dimension 6.')
    b = np.array([1, 4, 5, 4, 2, 1])
    C = [[9, 1, 7, 5, 4, 7], 
        [1, 11, 4, 2, 7, 5], 
        [7, 4, 13, 5, 0, 7], 
        [5, 2, 5, 17, 1, 9], 
        [4, 7, 0, 1, 21, 15], 
        [7, 5, 7, 9, 15, 27]]
    C = np.array(C)
    return 5 + b@x + x @ (C @ x)

def gradV_a(x):
    if x.shape[0] != 6:
        raise ValueError('X must have dimension 6.')
    b = np.array([1, 4, 5, 4, 2, 1])
    C = [[9, 1, 7, 5, 4, 7], 
        [1, 11, 4, 2, 7, 5], 
        [7, 4, 13, 5, 0, 7], 
        [5, 2, 5, 17, 1, 9], 
        [4, 7, 0, 1, 21, 15], 
        [7, 5, 7, 9, 15, 27]]
    C = np.array(C)
    return b + 2 * C @ x
#_________________________________________________


#-_____________COST FUNCTION B___________________
def V_b(x):
    if x.shape[0] != 2:
        raise ValueError('X must have dimension 2.')
    x1, x2 = x[0], x[1]
    num = ((x1**2 + 1)*(2*x2**2 + 1))**0.5
    den = x1**2 + x2**2 + 0.5
    return - num / den

def gradV_b(x):
    if x.shape[0] != 2:
        raise ValueError('X must have dimension 2.')
    x1, x2 = x[0],x[1]

    num = (-x1**3 + x1*x2**2 - 1.5*x1)*(2*x2**2+1)**0.5
    den = (x1**2 + x2**2 + 0.5)**2 * (x1**2 + 1)**0.5
    dx1 = -num / den

    num = (-2*x2**3 + 2*x2*x1**2 - x2)*(x1**2+1)**0.5
    den = (x1**2 + x2**2 + 0.5)**2 * (2*x2**2 + 1)**0.5
    dx2 = -num / den

    return np.array([dx1,dx2])
#_________________________________________________


#-_____________COST FUNCTION C___________________
def V_c(x):
    if x.shape[0] != 2:
        raise ValueError('X must have dimension 2.')
    x1, x2 = x[0], x[1]
    a = 1
    b = np.array([1,2])
    C = np.array([[12, 3],
                  [3, 10]])
    d = 10*np.log(1 + x1**4)*np.sin(100*x1)+10*np.log(1 + x2**4)*np.cos(100*x2)   
    V = a + b@x + 0.5*x @ (C @ x) + d
    return  V

def gradV_c(x):
    if x.shape[0] != 2:
        raise ValueError('X must have dimension 2.')
    x1, x2 = x[0], x[1]
    a = 1
    b = np.array([1,2])
    C = np.array([[12, 3],
                  [3, 10]])  
    grad =  b + C @ x
    grad_x1, grad_x2 = grad[0], grad[1]
    grad_x1 += 10 * ( (4*x1**3 * np.sin(100*x1) / (1+x1**4)) + 100*np.sin(100*x1)*np.log(1+x1**4) )
    grad_x2 += 10 * ( (4*x1**3 * np.sin(100*x1) / (1+x1**4)) - 100*np.cos(100*x1)*np.log(1+x1**4) )
    return np.array([grad_x1, grad_x2])


