from multiprocessing.sharedctypes import Value
import numpy as np
from numpy.linalg import norm, eig
from functools import partial



def steepest_descent(x0,
                     cost_function,
                     gradient_function,
                     step_size = 'armijo',
                     threshold = 1e-8, 
                     log = False, 
                     h = 1e-8, 
                     max_iter = 1e12, 
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
            should be close to 0. Default: 1e-8
        log :: bool
            True to log optimization progress. Default: False
        h :: float
            Parameter for finite difference estimation. 
            Default 1e-8
        max_iter :: int
            Maximum optimization iterations. Default: 1e12
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

    #iterate until near zero gradient or max iterations reached
    while norm(gradient) >= threshold and i <= max_iter: 

        #update gradient
        gradient = gradient_function(x)

        #determine step size
        if step_size == 'armijo':
            step_size = armijo(x, cost_function, gradient_function, gradient)
        elif not isinstance(step_size, (int, float)):
            raise ValueError('step size should be float, int or "armijo"')
        
        # move to a new x by moving from the original x in the negative
        # direction of the gradient according to a given step size
        x = x - step_size*gradient
        minimum = cost_function(x).item()

        #result tracking
        i += 1
        if log and i % 1e4 == 0: 
            print(f'x = {x.flatten()}, V(x) = {minimum:.5f}')
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
                     max_iter = 1e12, 
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
        max_iter :: int
            Maximum optimization iterations. Default: 1e12
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

    while norm(prev_gradient) >= threshold and i <= max_iter: 
        
        #determine step size
        if step_size == 'armijo':
            step_size = armijo(x, cost_function, gradient_function, prev_gradient)
        elif not isinstance(step_size, (int, float)):
            raise ValueError('step size should be float, int or "armijo"')

        #conjugate_gradient_algorithm
        x1 = x0 + step_size * search_direction
        next_gradient = gradient_function(x1)
        beta = (next_gradient - prev_gradient).T @  next_gradient
        beta /= prev_gradient.T @ prev_gradient
        search_direction = -1*next_gradient + beta * search_direction
        prev_gradient = next_gradient
        x0 = x1
        minimum = cost_function(x0).item()

        #track results
        if log and i%1e4 == 0: 
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
        e = np.eye(len(x))
        for i in range(x.shape[0]):
            grad = self.function(x + e[i].reshape(-1,1)*self.h) - self.function(x - e[i].reshape(-1,1)*self.h)
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
        e = np.eye(len(x))
        for i in range(x.shape[0]):
            grad = self.function(x + e[i].reshape(-1,1)*self.h) - self.function(x)
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
            raise ValueError("Method must be 'central' or 'forward'")

def cone_condition(g, s, theta=89):
    cos_phi = (-s.T @ g) / (np.linalg.norm(s) * np.linalg.norm(g))
    cos_theta = np.cos(theta * 2 * np.pi / 360)

    return cos_phi > cos_theta

def secant(x0,
            cost_function,
            gradient_function,
            H = None,
            step_size = 'armijo',
            threshold = 1e-8, 
            log = False, 
            h = 1e-8, 
            max_iter = 1e12, 
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
        max_iter :: int
            Maximum optimization iterations. Default: 1e12
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
    x_history, V_history = [], []

    while True:
        gradient_x0 = gradient_function(x0)
        s = -(H @ gradient_x0)

        if not cone_condition(gradient_x0, s):
            j = 0
            s = -gradient_function(x0)

        #determine step size
        if step_size == 'armijo':
            w = armijo(x0, cost_function, gradient_x0, search_dir=s.flatten(), gamma=1.5, r=0.8, log=False)
        elif isinstance(step_size, (int, float)):
            w = step_size
        else: 
            raise ValueError('step size should be float, int or "armijo"')
        
        
        x1 = x0 + w*s
        gradient_x1 = gradient_function(x1)

        if norm(gradient_x1) < threshold or j > max_iter:
            break

        dx = x1-x0
        dg = gradient_x1-gradient_x0
        H = H + np.matmul(dx,dx.T)/np.matmul(dx.T,dg) - np.matmul(np.matmul(H,dg),(np.matmul(H,dg)).T)/np.matmul(dg.T,np.matmul(H,dg))
        
        j += 1
        x0 = x1
        minimum = cost_function(x0).item()
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
    
    def v_bar(cost_x,grad_x_s,w):
        return cost_x + 0.5*w*grad_x_s

    w = 0.1
    cost_x = cost_function(x)
    grad_x_s = gradient @ search_dir
    # initialize p
    p = 0
    # propogate forward
    while cost_function(x + (gamma**p)*search_dir) < v_bar(cost_x, grad_x_s, (gamma**p)): 
        w = gamma**p
        # increment p
        p += 1
    # initialize q
    q = 0
    # propogate backwards
    while cost_function(x + (r**q * gamma**p)*search_dir) > v_bar(cost_x, grad_x_s, r**q * gamma**p): 
        # consider step size w
        w = r**q * gamma**p
        # increment q
        q += 1
    # return step size
    if log:
        print(f'p={p}, q={q}, w={w}')
    return w

def penalty_fn(x0,
               cost_function,
               gradient_function,
               step_size='armijo',
               ecp=None,
               icp=None,
               threshold=1e-6,
               log = False, 
               h = 1e-8, 
               max_iter = 1e12, 
               fd_method = 'central', 
               track_history = False):
    
    def phi(cost_function, sigma, ecp, icp, x):
        cost = cost_function(x)
        if ecp is not None:
            cost = cost + 0.5*sigma*norm(ecp(x))**2
        if icp is not None:
            for eq in icp:
                cost += 0.5*sigma*norm(np.minimum(eq(x),np.zeros_like(eq(x))))**2
        return cost
    
    def cost_norm(x):
        cost = 0
        if ecp is not None:
            cost = cost + norm(ecp(x))**2
        if icp is not None:
            for eq in icp:
                cost += norm(np.minimum(eq(x),np.zeros_like(eq(x))))**2
        return np.sqrt(cost)

    sigma = 1
    x = x0

    while cost_norm(x) > threshold:
        x, _ = steepest_descent(x0,
                             partial(phi, cost_function, sigma, ecp, icp),
                             gradient_function,
                             step_size,
                             1e-3,
                             log,
                             h,
                             max_iter,
                             fd_method,
                             track_history)
        sigma *= 10
        if sigma >= 1e5:
            break
    return x, cost_function(x).item()

