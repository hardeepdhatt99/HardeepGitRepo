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

def gauss_elimination(A, b):
    """Return a vector x with np.matmul(A, x) == b using
    the Gauss Elimination algorithm, without partial pivoting."""
    for k in range(A.shape[0] - 1):
        eliminate(A, b, k)
    x = backward_substitution(A, b)
    return x

e = pow(2, -100)
A = np.array([[e, 1],
              [1, 1]])
b = np.array([1 + e, 2])

soln_nopivot = gauss_elimination(A, b)

def partial_pivot(A, b, k):
    """Perform partial pivoting for column k. That is, swap row k
    with row j > k so that the new element at A[k,k] is the largest
    amongst all other values in column k below the diagonal.
    
    This function should modify A and b in place.
    """
    biggest = math.fabs(A[k, k])
    index = k
    for i in range(k, A.shape[0]):
        if math.fabs(A[i, k]) > biggest:            
            biggest = math.fabs(A[i, k])
            index = i
    A[[k, index]] = A[[index, k]]
    b[k], b[index] = b[index], b[k]
        

def gauss_elimination_partial_pivot(A, b):
    """Return a vector x with np.matmul(A, x) == b using
    the Gauss Elimination algorithm, with partial pivoting."""
    for k in range(A.shape[0] - 1):
        partial_pivot(A, b, k)
        eliminate(A, b, k)
    x = backward_substitution(A, b)
    return x

e = pow(2, -100)
A = np.array([[e, 1],
              [1, 1]])
b = np.array([1 + e, 2])

soln_pivot = gauss_elimination_partial_pivot(A, b)


def norm_1(M):
    b = np.absolute(M.transpose())
    norm = 0
    for i in range(M.shape[0]):
        s = sum(b[i])
        if s > norm:
            norm = s
    return norm

def inverse_2x2(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return np.array([[M[1][1]/det, -1*M[0][1]/det],
                    [-1*M[1][0]/det, M[0][0]/det]])

def matrix_condition_number(M):
    """
    Returns the condition number of the 2x2 matrix M.
    Use the $L_1$ matrix norm.

    Precondition: M.shape == [2, 2] 
                  M is non-singular

    >>> matrix_condition_number(np.array([[1., 0.], [0., 1.]]))
    1
    """
    return norm_1(M) * norm_1(inverse_2x2(M))