import math
import numpy as np

def backward_substitution(A, b):
    """Return a vector x with np.matmul(A, x) == b, where 
        * A is an nxn numpy matrix that is upper-triangular and non-singular
        * b is an nx1 numpy vector
    """
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float)
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(n-1, i, -1):
            s += A[i,j] * x[j]
        x[i] = (b[i] - s) / A[i,i]
    return x

def eliminate(A, b, k):
    """Eliminate the k-th row of A, in the system np.matmul(A, x) == b,
    so that A[i, k] = 0 for i < k. The elimination is done in place."""
    n = A.shape[0]
    for i in range(k + 1, n):
        m = A[i, k] / A[k, k]
        for j in range(k, n):
            A[i, j] = A[i, j] - m * A[k, j]
        b[i] = b[i] - m * b[k]
        
def elim_matrix(A, k):
    """Eliminate the k-th row of a matrix"""
    n = A.shape[0]
    for i in range(k + 1, n):
        m = A[i, k] / A[k, k]
        for j in range(k, n):
            A[i, j] = A[i, j] - m * A[k, j]
        

def gauss_elimination(A, b):
    """Return a vector x with np.matmul(A, x) == b using
    the Gauss Elimination algorithm, without partial pivoting."""
    for k in range(A.shape[0] - 1):
        eliminate(A, b, k)
    x = backward_substitution(A, b)
    return x


def forward_substitution(A, b):
    """Return a vector x with np.matmul(A, x) == b, where 
        * A is an nxn numpy matrix that is lower-triangular and non-singular
        * b is a nx1 numpy vector
        
    >>> A = np.array([[2., 0.],
                      [1., -2.]])
    >>> b = np.array([1., 2.])
    >>> forward_substitution(A, b)
    array([ 0.5 , -0.75])
    """
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float)
    for i in range(0, n, 1):
        s = 0
        for j in range(0, i, 1):
            s += A[i,j] * x[j]
        x[i] = (b[i] - s) / A[i,i]
    return x


def elementary_elimination_matrix(A, k):
    """Return the elements below the k-th diagonal of the
    k-th elementary elimination matrix.    

    (Do not use partial pivoting, since we haven't
    introduced the idea yet.)
    
    Precondition: A is an nxn numpy matrix, non-singular
                  A[i,j] = 0 for i > j, j < k - 1

    As always, these examples are for your understanding only.
    The actual Python output might differ slightly.
    
    >>> 
    >>> elementary_elimination_matrix(A, 1)
    np.array([[1., 0., 0.],
              [-0.5, 1., 0.],
              [-1., 0., 1.]])
    >>> A = np.array([[2., 0., 1.],
                      [0., 1., 0.],
                      [0., 1., 2.]])
    >>> elementary_elimination_matrix(A, 2)
    np.array([[1., 0., 0.],
              [0., 1., 0.],
              [0., -1., 1.]])
    """
    n = A.shape[0]
    k = k-1
    M = np.eye(n)
    for i in range(k + 1, n, 1):
        s = (A[i, k] / A[k, k]) * -1
        M[i, k] = s
    
    return M
    


def lu_factorize(A):
    """Return two matrices L and U, where
            * L is lower triangular
            * U is upper triangular
            * and np.matmul(L, U) == A
    >>> A = np.array([[2., 0., 1.],
                      [1., 1., 0.],
                      [2., 1., 2.]])
    >>> L, U = lu_factorize(A)
    >>> L
    array([[1. , 0. , 0. ],
           [0.5, 1. , 0. ],
           [1. , 1. , 1. ]])
    >>> U
    array([[ 2. ,  0. ,  1. ],
           [ 0. ,  1. , -0.5],
           [ 0. ,  0. ,  1.5]])
    """
    n = A.shape[0]
    L = np.zeros_like(n, dtype=np.float)
    U = A
    
    for k in range(1, n):
        s = elementary_elimination_matrix(A, k)        
        L = np.add(L, s)
    
    L = np.multiply(L,-1.0)
    np.fill_diagonal(L, 1)
    
    for i in range(n - 1):
        elim_matrix(A, i)
        
    return L, A


def solve_lu(A, b):
    """Return a vector x with np.matmul(A, x) == b using
    LU factorization. (Do not use partial pivoting, since we haven't
    introduced the idea yet.)
    """
    L, U = lu_factorize(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x
    


def invert_matrix(A):
    """Return the inverse of the nxn matrix A by 
    solving n systems of linear equations of the form 
    Ax = b.
    
    >>> A = np.array([[0.5, 0.],
                      [-1., 2.]])
    >>> invert_matrix(A)
    array([[ 2. , -0. ],
           [ 1. ,  0.5]])
    """
   
    n = A.shape[0]
    ident = np.eye(n)
    inv = []
    for i in range(0, n, 1):        
        inv.append(solve_lu(A, ident[i]))
    
    return np.column_stack(tuple(inv))