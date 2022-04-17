from multiprocessing.sharedctypes import Value
import numpy as np
from numpy.linalg import norm, eig



def steepest_descent(x0,
                     cost_function,
                     gradient_function,
                     step_size = 'armijo',
                     threshold = 1e-4, 
                     log = False, 
                     h = 1e-8, 
                     max_iter = 1e6, 
                     gamma = 1.5, 
                     r = 0.8, 
                     fd_method = 'central', 
                     track_history = False):
    '''
    Performs vanilla gradient descent. 
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo
        threshold :: float
            Threshold at which to stop minimization. Values 
            should be close to 0. Default: 1e-4
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        gamma :: float
            Gamma parameter for armijo. Default is 1.5. 
        r :: float
            r parameter for armijo. Default is 0.8. 
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    '''

    #if no gradient function available, use finite difference appx
    if gradient_function == None: 
        fd = Finite_Difference(cost_function, fd_method, h)
        gradient_function = fd.estimate_gradient

    x_history, V_history = [],[]
    #initialize iterator, x, and gradient
    i = 0
    x = x0
    gradient = gradient_function(x)
    minimum = cost_function(x)

    #iterate until near zero gradient or max iterations reached
    while norm(gradient) >= threshold and i <= max_iter: 

        #update gradient
        gradient = gradient_function(x)
        search_dir = -1*gradient

    #determine step size
        if step_size == 'armijo':                      
            w = armijo(x, cost_function, gradient, search_dir=search_dir.flatten(), gamma = gamma, r = r, log=False)
        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise TypeError('step size should be float, int or "armijo"') 
        
        # move to a new x by moving from the original x in the negative
        # direction of the gradient according to a given step size
        x = x + w*search_dir.flatten()
        minimum = cost_function(x)

        #result tracking
        i += 1
        if log and i % 1e4 == 0: 
            print(f'x = {x}, V(x) = {minimum:.5f}')
        if track_history: 
            x_history.append(x), V_history.append(minimum)

    if track_history:
        return x, minimum, x_history, V_history
    else: 
        return x, minimum





def conjugate_gradient(x0,
                     cost_function,
                     gradient_function,
                     step_size,
                     threshold = 1e-8, 
                     log = False, 
                     h = 1e-8, 
                     max_iter = 1e6, 
                     gamma = 1.5, 
                     r = 0.8, 
                     fd_method = 'central', 
                     track_history = False):
    '''
    Performs conjugate gradient descent. 
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo
        threshold :: float
            Threshold at which to stop minimization. Values 
            should be close to 0. Default: 1e-8
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        gamma :: float
            Gamma parameter for armijo. Default is 1.5. 
        r :: float
            r parameter for armijo. Default is 0.8. 
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    '''
    #if no gradient function available, use finite difference appx
    if gradient_function == None: 
        fd = Finite_Difference(cost_function, fd_method, h)
        gradient_function = fd.estimate_gradient

    x_history, V_history = [],[]    
    i = 0
    prev_gradient = gradient_function(x0)
    search_direction = prev_gradient * -1
    minimum = cost_function(x0)
    while norm(prev_gradient) >= threshold and i <= max_iter: 

        #determine step size
        if step_size == 'armijo':
            w = armijo(x0,
                    cost_function,
                    prev_gradient,
                    search_dir=search_direction.flatten(),
                    gamma = gamma,
                    r = r,
                    log=False)

        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise TypeError('step size should be float, int or "armijo"')

        #conjugate_gradient_algorithm
        x1 = x0 + w * search_direction
        next_gradient = gradient_function(x1)
        beta = (next_gradient - prev_gradient) @  next_gradient
        beta /= prev_gradient @ prev_gradient.transpose()
        search_direction = -1*next_gradient + beta * search_direction
        prev_gradient = next_gradient
        x0 = x1
        minimum = cost_function(x1)

        #track results
        if log and i%1e4 == 0: 
            print(f'x = {x1}, V(x) = {minimum:.5f}')
        if track_history: 
            x_history.append(x1), V_history.append(minimum)

    if track_history:
        return x1, minimum, x_history, V_history
    else: 
        return x1, minimum

def secant(x0,
            cost_function,
            gradient_function,
            H = None,
            step_size = 'armijo',
            threshold = 1e-8, 
            log = False, 
            h = 1e-8, 
            max_iter = 1e6, 
            gamma = 1.5, 
            r = 0.8,
            fd_method = 'central', 
            track_history = False):
    '''
    Performs minimization using secant algorithm with Davidson-Fletcher-Powell.  
    Args: 
        x0 :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        H :: np.array (shape: len(x0) x len(x0))
            Estimate for C inverse. Default is None.
            if None, H = eye(len(x0)) is used. 
        step_size :: float or String
            Step size to use during gradient descent. 
            If 'armijo', Armijo step size selection is used. 
            Default: 'armijo
        threshold :: float
            Threshold at which to stop minimization. Values 
            should be close to 0. Default: 1e-8
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        gamma :: float
            Gamma parameter for armijo. Default is 1.5. 
        r :: float
            r parameter for armijo. Default is 0.8. 
        max_iter :: int
            Maximum optimization iterations. Default: 1e6
        fd_method :: string
            Method for finite difference estimation. 
            Options: 'central', 'forward'
        track_history :: bool
            True to track points visited and corresponding cost. 
    Returns: 
        x :: np.array
            Point at which minimization is reached. Shape (n,)
        minimum :: float
            Value of cost function at optimizer. 
        x_history :: list
            List of points visisted. (if track_history = True)
        V_history :: list
            List of costs visisted. (if track_history = True)
    '''
    if H == None:
        H = np.eye(len(x0))
    if H.shape[0] != H.shape[1] != len(x0):
        raise ValueError('H should be square numpy array with n = len(x0).')

    #if no gradient function available, use finite difference appx
    if gradient_function == None: 
        fd = Finite_Difference(cost_function, fd_method, h)
        gradient_function = fd.estimate_gradient

    j=0
    x_history, V_history = [],[]
    minimum = cost_function(x0)
    while True:
        gradient_x0 = gradient_function(x0)
        s = -np.matmul(H,gradient_x0.reshape(-1,1))
        
        #determine step size
        if step_size == 'armijo':                      
            w = armijo(x0, cost_function, gradient_x0, search_dir=s.flatten(), gamma = gamma, r = r, log=False)
        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise TypeError('step size should be float, int or "armijo"') 
        x1 = x0 + w*s.flatten()
        gradient_x1 = gradient_function(x1)
        if norm(gradient_x1) < threshold or j > max_iter:
            break
        dx = (x1-x0).reshape(-1,1)
        dg = (gradient_x1-gradient_x0).reshape(-1,1)
        H = H + np.matmul(dx,dx.reshape(1,-1))/np.dot(dx.reshape(1,-1),dg) - np.matmul(np.matmul(H,dg),(np.matmul(H,dg)).reshape(1,-1))/np.dot(dg.reshape(1,-1),np.matmul(H,dg))
        j += 1
        x0 = x1
        minimum  = cost_function(x0)

        x_history.append(x0), V_history.append(minimum)

        #track results
        if log and j%1e4 == 0: 
            print(f'x = {x0}, V(x) = {minimum:.5f}')
        if track_history: 
            x_history.append(x0), V_history.append(minimum)

    if track_history:
        return x0, minimum, x_history, V_history
    else: 
        return x0, minimum


class Finite_Difference:
    def __init__(self, function, method, h = 1e-8):
        '''
        Args: 
            function: cost function Rn -> R
            h: Default is 1e-8. 
        '''
        self.function = function
        self.h = h
        self.method = method

    def central_difference(self, x):
        '''
        Performs central difference estimate of the gradient. 
        Args: 
            x: np.array
                Point at which to estimate derivative. Shape (n,)
        Returns: 
            gradient: np.array
                Gradient estimate at x. Shape (n,)
        '''
        gradient = np.zeros_like(x)
        for i in range(x.shape[0]):
            e = np.eye(1,x.shape[0],i).squeeze()
            grad = self.function(x + e*self.h) - self.function(x - e*self.h)
            grad /= 2*self.h
            gradient[i] = grad
        return gradient

    def forward_difference(self, x):
        '''
        Performs forward difference estimate of the gradient. 
        Args: 
            x: np.array
                Point at which to estimate derivative. Shape (n,)
        Returns: 
            gradient: np.array
                Gradient estimate at x. Shape (n,)
        '''
        gradient = np.zeros_like(x)
        for i in range(x.shape[0]):
            e = np.eye(1,x.shape[0],i).squeeze()
            grad = self.function(x + e*self.h) - self.function(x)
            grad /= self.h
            gradient[i] = grad
        return gradient  

    def estimate_gradient(self, x):
        '''
        Select finite difference method
        '''
        if self.method == 'central':
            return self.central_difference(x)
        elif self.method == 'forward': 
            return self.forward_difference(x)
        else: 
            raise TypeError("Method must be 'central' or 'forward'")





def armijo(x,
           cost_function,
           gradient,
           search_dir,
           gamma=1.5,
           r = 0.8,
           log=True):
    '''
    Determine step size using secant algorithm
    Args: 
        x :: np.array
            Initial point of minization. Shape (n,)
        cost_function :: Python function
            Cost function to minimize. Rn -> R. 
        gradient_function :: Python function, or None
            Gradient of cost_function. Rn -> Rn
            If None, finite difference estimation of gradient is used. 
        search_dir :: np.array (Shape (n,))
            Search direction vector. 
        gamma :: float
            Gamma parameter for armijo algorithm. Default: 1.5
        r :: 0.8
            r parameter for armijo algorithm. Default: 0.8
        log :: bool
            True to log armijo optimization progress. Default: False 
    Returns: 
        w :: float
            Optimal step size at x in direction search_dir. 
    '''
    
    def v_bar(w):
        return cost_x + 0.5*w*grad_x_s

    w = 1
    cost_x = cost_function(x)
    grad_x_s = np.dot(gradient,search_dir)
    # initialize p
    p = 0
    # propogate forward
    w = gamma**p
    while cost_function(x + w*search_dir) < v_bar(w): 
        w = gamma**p
        # increment p
        p += 1
    # initialize q
    q = 0
    # propogate backwards
    w = r**q * gamma**p
    while cost_function(x + w*search_dir) > v_bar(w): 
        # increment q
        q += 1
        # consider step size w
        w = r**q * gamma**p

    # return step size
    if log:
        print(f'p={p}, q={q}, w={w}')
    return w


# def armijo(x, cost_function, gradient, s, gamma=1.5, mu=0.8):
#     """Armijo algorithm for computing step size

#     Parameters
#     ----------
#     gamma : float
#         Parameter for increasing step size
#     mu : float
#         Parameter for decreasing step size

#     Returns
#     -------
#     float
#         Step size
#     """

#     w = 1  # Default step size

#     k_g = 0  # Power of gamma
#     k_m = 0  # Power of mu

#     # Precompute cost and gradient to save time
#     vx = cost_function(x)
#     gx_s = gradient @ s

#     def v_bar(w):
#         return vx + 0.5 * w * gx_s

#     while p.cost(x + gamma**k_g * s) < v_bar(gamma**k_g):
#         k_g += 1
#         w = gamma**k_g

#     while p.cost(x + mu**k_m * gamma**k_g * s) > v_bar(mu**k_m * gamma**k_g):
#         k_m += 1
#         w = mu**k_m * gamma**k_g

#     return w
