import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product

def func(x):
    return x**2 - 2*x - 3

def fprime(x):
    return 2*x - 2

def plotFunc(x0):
    x = np.linspace(-5, 7, 100)
    plt.plot(x, func(x))
    plt.plot(x0, func(x0), 'ro')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('Objective Function')
    plt.show()

def plotPath(xs, ys, x0):
    plotFunc(x0)
    plt.plot(xs, ys, linestyle='--', marker='o', color='orange')
    plt.plot(xs[-1], ys[-1], 'ro')


x0 = -4
plotFunc(x0)

def GradientDescentSimple(func, fprime, x0, alpha, tol=1e-5, max_iter=1000):
    # initialize x, f(x), and -f'(x)
    xk = x0
    fk = func(xk)
    pk = -fprime(xk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    # take steps
    while abs(pk) > tol and num_iter < max_iter:
        # calculate new x, f(x), and -f'(x)
        xk = xk + alpha * pk
        fk = func(xk)
        pk = -fprime(xk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
    # print results
    if num_iter == max_iter:
        print('Gradient descent does not converge.')
    else:
        print('Solution found:\n  y = {:.4f}\n  x = {:.4f}'.format(fk, xk))
    
    return curve_x, curve_y

xs, ys = GradientDescentSimple(func, fprime, x0, alpha=0.1)
plotPath(xs, ys, x0)

def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.
    
    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    pk : array
        Search direction.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.
    
    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0*pk)
    
    while not phi_a0 <= phi0 + c1*alpha0*derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0*pk)
    
    return alpha0, phi_a0

def Griewank(xs):
    """attempt to calculate the length required"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)
    
    """If it does not fit address length it will crash here with EXEC error"""
    sigma = np.dot(xs, xs) / 4000
    pi = np.prod(cos_terms)
    return 1 + sigma - pi

def GriewankGrad(xs):
    """First derivative of Griewank Function"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)
    pi_coefs = np.prod(cos_terms) / cos_terms
    
    sigma = 2 * xs / 4000
    pi = pi_coefs * np.sin(xs / sqrts) * (1 / sqrts)
    return sigma + pi

x = np.arange(-5, 5, 0.025)
y = np.arange(-5, 5, 0.025)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
mesh_size = range(len(X))
for i, j in product(mesh_size, mesh_size):
    x_coor = X[i][j]
    y_coor = Y[i][j]
    Z[i][j] = Griewank(np.array([x_coor, y_coor]))

fig = plt.figure(figsize=(6,6))
ax = fig.gca()
ax.set_title('2D Griewank Function')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

def GradientDescent(f, f_grad, init, alpha=1, tol=1e-5, max_iter=1000):
    """Gradient descent method for unconstraint optimization problem.
    given a starting point x ∈ Rⁿ,
    repeat
        1. Define direction. p := −∇f(x).
        2. Line search. Choose step length α using Armijo Line Search.
        3. Update. x := x + αp.
    until stopping criterion is satisfied.
    
    Parameters
    --------------------
    f : callable
        Function to be minimized.
    f_grad : callable
        The first derivative of f.
    init : array
        initial value of x.
    alpha : scalar, optional
        the initial value of steplength.
    tol : float, optional
        tolerance for the norm of f_grad.
    max_iter : integer, optional
        maximum number of steps.
    
    Returns
    --------------------
    xs : array
        x in the learning path
    ys : array
        f(x) in the learning path
    """
    # initialize x, f(x), and f'(x)
    xk = init    
    fk = f(xk)
    gfk = f_grad(xk)
    gfk_norm = np.linalg.norm(gfk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    print('Initial condition: y = {:.4f}, x = {} \n'.format(fk, xk))
    # take steps
    while gfk_norm > tol and num_iter < max_iter:
        # determine direction
        pk = -gfk
        # calculate new x, f(x), and f'(x)
        alpha, fk = ArmijoLineSearch(f, xk, pk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * pk
        gfk = f_grad(xk)
        gfk_norm = np.linalg.norm(gfk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
              format(num_iter, fk, xk, gfk_norm))
        
def plot(xs, ys):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle('Gradient Descent Method')

    ax1.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
    ax1.plot(xs[-1,0], xs[-1,1], 'ro')
    ax1.set(
        title='Path During Optimization Process',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax1.contour(X, Y, Z)
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    ax2.plot(ys, linestyle='--', marker='o', color='orange')
    ax2.plot(len(ys)-1, ys[-1], 'ro')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(
        title='Objective Function Value During Optimization Process',
        xlabel='Iterations',
        ylabel='Objective Function Value'
    )
    ax2.legend(['Armijo line search algorithm'])

x0 = np.array([0, 3])
xs, ys = GradientDescent(Griewank, GriewankGrad,init=x0)
plot(xs, ys)
