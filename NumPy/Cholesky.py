import math
import numpy as np

def cholesky_factorize(A):
    """Return the Cholesky Factorization L of A, where
        * A is an nxn symmetric, positive definite matrix
        * L is lower triangular, with positive diagonal entries
        * $A = LL^T$
        
    >>> M = np.array([[8., 3., 2.],
                      [3., 5., 1.],
                      [2., 1., 3.]])
    >>> L = cholesky_factorize(M)
    >>> np.matmul(L, L.T)
    array([[8., 3., 2.],
           [3., 5., 1.],
           [2., 1., 3.]])
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j] * L[k,j] for j in range(k))
            if(i==k):
                L[i,k] = math.sqrt(A[i,i] - s)
            else:
                L[i,k] = 1.0 / L[k,k] * (A[i,k] - s)
    return L
        
    
def backward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float)
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(n-1, i, -1):
            s += A[i,j] * x[j]
        x[i] = (b[i] - s) / A[i,i]
    return x

def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float)
    for i in range(0, n, 1):
        s = 0
        for j in range(0, i, 1):
            s += A[i,j] * x[j]
        x[i] = (b[i] - s) / A[i,i]
    return x

def solve_lu(L, U, b):
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def solve_rank_one_update(L, U, b, u, v):
    """Return the solution x to the system (A - u v^T)x = b, where
    A = LU, using the approach we derived in class using
    the Sherman Morrison formula. You may assume that
    the LU factorization of A has already been computed for you, and
    that the parameters of the function have:
        * L is an invertible nxn lower triangular matrix
        * U is an invertible nxn upper triangular matrix
        * b is a vector of size n
        * u and b are also vectors of size n

    >>> A = np.array([[2., 0., 1.],
                      [1., 1., 0.],
                      [2., 1., 2.]])
    >>> L, U = lu_factorize(A) # from homework 3
    >>> L
    array([[1. , 0. , 0. ],
           [0.5, 1. , 0. ],
           [1. , 1. , 1. ]])
    >>> U
    array([[ 2. ,  0. ,  1. ],
           [ 0. ,  1. , -0.5],
           [ 0. ,  0. ,  1.5]])
    >>> b = np.array([1., 1., 0.])
    >>> u = np.array([1., 0., 0.])
    >>> v = np.array([0., 2., 0.])
    >>> x = solve_rank_one_update(L, U, b, u, v)
    >>> x
    array([1. , 0. , -1.])
    >>> np.matmul((A - np.outer(u, v)), x)
    array([1. , 1. , 0.])
    """
    y = solve_lu(L, U, b)
    z = solve_lu(L, U, u)    
    vt = np.transpose(v)
    vtz = np.matmul(vt, z)
    vty = np.matmul(vt,y)
    scal = vty / (1 - vtz)
    return y + (scal * z)


def run_example():
    A = np.array([[2., 0., 1.],
                  [1., 1., 0.],
                  [1., 1., 1.]])
    L = np.array([[1., 0., 0.],
                  [0.5, 1., 0.],
                  [0.5, 1., 1.]])
    U = np.array([[2., 0., 1.],
                  [0., 1., -0.5],
                  [0., 0., 1.]])
    b = np.array([1, 1, -1])
    u = np.array([0, 0, 0.9999999999999999])
    v = np.array([0, 0, 0.9999999999999999])
    x = solve_rank_one_update(L, U, b, u, v)
    print(np.matmul((A - np.outer(u, v)), x) - b)


def solve_rank_one_update_iterative(L, U, b, u, v, x):
    """Return a better solution x* to the system (A - u v^T)x = b,
    where A = LU. The first 5 parameters are the same as those of the 
    function `solve_rank_one_update`. The last parameter is an 
    estimate `x` of the solution.

    This function should perform exactly *one* iterative refinement
    iteration.
    """
    A = np.matmul(L,U)
    r0 = b - np.matmul(A, x)
    z0 = solve_rank_one_update(L, U, r0, u, v)
    x1 = x + z0
    return solve_rank_one_update(L, U, x1, u, v)