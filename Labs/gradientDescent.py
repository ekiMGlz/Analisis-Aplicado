import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la

def grad(f, x):
    '''
        Input:
            f: lambda function
            x: function args
        Output:
            grad_f: function gradient at x
    '''
    n = len(x)
    grad_f = np.zeros(n)
    
    E = np.diag([pow(np.finfo(float).eps, 1/3) * (abs(a) + 1) for a in x])
    
    for i in range(n):
        grad_f[i] = (f(x + E[:, i]) - f(x - E[:, i])) * (0.5 / E[i, i])
    
    return grad_f
    
def hess(f, x):
    '''
        Input:
            f: lambda function
            x: function args
        Output:
            hess_f: hessian of f at x
    '''
    n = len(x)
    hess_f = np.zeros([n, n])
    
    E = np.diag([pow(np.finfo(float).eps, 1/4) * (abs(a) + 1) for a in x])
    
    for i in range(n):
        for j in range(n):
            hess_f[i, j] = (  f(x + E[:, i] + E[:, j]) 
                            - f(x - E[:, i] + E[:, j]) 
                            - f(x + E[:, i] - E[:, j]) 
                            + f(x - E[:, i] - E[:, j]) ) * (0.25 / (E[i, i] * E[j, j]))
    
    return hess_f

def coor_descent(f, x0, tol=1e-5, maxiter=100):
    '''
        Input:
            f - func to minimize
            x0 - initial point
            tol - tolerance
            maxiter - max num of iterations
        Output:
            xf - final aproximation of x*
            iterations - number of iterations to reach soln
    '''
    
    xf = x0
    n = len(x0)
    iterations = 0
    c_1 = 0.1
    
    d = np.zeros(n)
    grad_f = grad(f, xf)
    
    while iterations < maxiter and np.linalg.norm(grad_f, np.inf) > tol:
        
        
        
        #Find descent direction
        i = np.argmax(abs(grad_f))
        d[i] = -np.sign(grad_f[i])
        
        #Direction coef.
        alfa = 1
        
        while f(xf + alfa*d) > f(xf) - alfa*c_1*grad_f[i]:
            alfa /= 2
        
        
        #Next iteration
        xf = xf + alfa*d
        iterations += 1
        d[i] = 0
        grad_f = grad(f, xf)
    
    return xf, iterations

def max_descent(f, x0, tol=1e-5, maxiter=100):
    '''
        Input:
            f - func to minimize
            x0 - initial point
            tol - tolerance
            maxiter - max num of iterations
        Output:
            xf - final aproximation of x*
            iterations - number of iterations to reach soln
    '''
    
    xf = x0
    n = len(x0)
    iterations = 0
    c_1 = 0.1
    
    grad_f = grad(f, xf)
    
    while iterations < maxiter and np.linalg.norm(grad_f, np.inf) > tol:
        
        #Find descent direction
        d = - grad_f / np.linalg.norm(grad_f)
        
        #Direction coef.
        alfa = 1
        
        while f(xf + alfa*d) > f(xf) + alfa*c_1*np.dot(grad_f, d):
            alfa /= 2
        
        #Next iteration
        xf = xf + alfa*d
        iterations += 1
        grad_f = grad(f, xf)
    
    return xf, iterations

def newton_descent(f, x0, tol=1e-5, maxiter=100):
    '''
        Input:
            f - func to minimize
            x0 - initial point
            tol - tolerance
            maxiter - max num of iterations
        Output:
            xf - final aproximation of x*
            iterations - number of iterations to reach soln
    '''
    
    xf = x0
    n = len(x0)
    iterations = 0
    c_1 = 0.1
    
    grad_f = grad(f, xf)
    hess_f = hess(f, xf)
    while iterations < maxiter and np.linalg.norm(grad_f, np.inf) > tol:
        
        #Find descent direction
        d = np.linalg.solve(hess_f, -grad_f)
        
        #Direction coef.
        alfa = 1
        
        while f(xf + alfa*d) > f(xf) + alfa*c_1*np.dot(grad_f, d):
            alfa /= 2
        
        #Next iteration
        xf = xf + alfa*d
        iterations += 1
        grad_f = grad(f, xf)
        hess_f = hess(f, xf)
    
    return xf, iterations

def trust_region(f, x0, r, min_quality, max_r=np.inf, tol=1e-5, maxiter=100):
    
    i = 0
    xk = x0
    fk = f(xk)
    gk = grad(f, xk)
    Bk = grad(f, xk)
    
    while i > maxiter and np.linalg.norm(grad_f, np.inf) > tol:
    
        mk = lambda d : fk + np.dot(gk, d) + 0.5 * np.dot(d, np.dot(Bk, d))
        dk = cauchy_point(gk, Bk, r)
        quality = (fk - f(xk + dk))/(fk - m(dk))

        if quality < 0.25:
            r = 0.25 * r
        elif quality > 0.75 and r - np.norm(dk) < 1e-10 :
            r = min(2*r, max_r)

        if quality > min_quality:
            xk += dk
            fk = f(xk)
            gk = grad(f, xk)
            Bk = grad(f, xk)
            
        i += 1
        
    return xk

def cauchy_point(gk, Bk, r):
    gk_norm = np.linalg.norm(gk) 
    p = -(r/gk_norm) * gk
    
    pBp = np.dot(p, np.dot(Q, p)) 
    
    alpha = min(1, r * gk_norm / pBp) if pBp > 0 else 1
    
    p *= alpha
    
    return p