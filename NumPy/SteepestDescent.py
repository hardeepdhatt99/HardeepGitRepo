#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np


def golden_section_search(f, a, b, n=10):
    """ Return the Golden Section search intervals generated in 
    an attempt to find a minima of a function `f` on the interval
    `[a, b]`. The returned list should have length `n+1`.
   
    Do not evaluate the function `f` more times than necessary.
    
    Example: (as always, these are for illustrative purposes only)

    >>> golden_section_search(lambda x: x**2 + 2*x + 3, -2, 2, n=5)
    [(-2, 2),
     (-2, 0.4721359549995796),
     (-2, -0.4721359549995796),
     (-1.4164078649987384, -0.4721359549995796),
     (-1.4164078649987384, -0.8328157299974766),
     (-1.1934955049953735, -0.8328157299974766)]
    """
    ret = [(a,b)]
    t = (math.sqrt(5) - 1) / 2
    x1 = a + ( 1 - t ) * ( b - a )
    f1 = f(x1)
    x2 = a + t * ( b - a )
    f2 = f(x2)
    for i in range(n):
        print("f(x1): " + str(f1))
        print("f(x2): " + str(f2))
        if(f1 > f2):
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + t*(b-a)
            f2 = f(x2)
            ret.append((a,b))
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1-t)*(b-a)
            f1 = f(x1)
            ret.append((a,b))
    return ret

golden_section_search(lambda x: math.pow((x-2), 2) * (1 + math.sin(x)), -2, 0, n=2)

def f1(x):
    return 2 * math.pow(x, 2) - 2 * x + 3

def f2(x):
    return -1 * x * math.pow(math.e, math.pow(-1*x, 2))

golden_f1 = golden_section_search(f1, 0, 2, 10)
golden_f2 = golden_section_search(f2, 0, 2, 10)


def newton_1d(f, df, ddf, x, n=10):
    """ Return the list of iterates computed when using 
    Newton's Method to find a local minimum of a function `f`,
    starting from the point `x`. The parameter `df` is the 
    derivative of `f`. The parameter `ddf` is the second 
    derivative of `f`. The length of the returned list 
    should be `n+1`.
    
    Example: (as always, these are for illustrative purposes only)
    
    >>> newton_1d(lambda x: x**3, 
    ...           lambda x: 3 * (x**2),
    ...           lambda x: 6 * x, 
    ...           x=1.0, n=5)
    [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    """
    ret = [x]
    xk = x
    for i in range(n):
        if(ddf(xk) == 0):
            break
        xk1 = xk - df(xk) / ddf(xk)
        xk = xk1
        ret.append(xk)
    return ret


def df1(x):
    return 4*x -2
def ddf1(x):
    return 4

def df2(x):
    return -1 * math.pow(math.e, math.pow(-1*x, 2)) + 2 * math.pow(math.e, math.pow(-1*x, 2)) * math.pow(x, 2)
def ddf2(x):
    return -4 * math.pow(math.e, math.pow(-1*x, 2)) * math.pow(x, 3) + 6 * math.pow(math.e, math.pow(-1*x, 2)) * x


newton_f1 = newton_1d(f1, df1, ddf1, 1, 30)
newton_f2 = newton_1d(f2, df2, ddf2, 1, 30)


def gradF1(x0, x1):
    return 8 * math.pow(x0, 3) - 2 * x0 * math.pow(x1, 2) - 2 * x0

def gradF2(x0, x1):
    return 12 * math.pow(x1, 3) - 2 * math.pow(x0, 2) * x1 - 8 * x1

def steepest_descent_f(init_x0, init_x1, alpha, n=5):
    """ Return the $n$ steps of steepest descent on the function 
    f(x_0, x_1) given in part(a), starting at (init_x0, init_x1).
    The returned value is a list of tuples (x_0, x_1) visited
    by the steepest descent algorithm, and should be of length
    n+1. The parameter alpha is used in place of performing
    a line search.
    
    Example:
    
    >>> steepest_descent_f(0, 0, 0.5, n=0)
    [(0, 0)]
    """
    ret = [(init_x0, init_x1)]
    a = init_x0
    b = init_x1
    for i in range(n):
        a1 = a - alpha * gradF1(a, b)
        b1 = b - alpha * gradF2(a, b)
        ret.append((a1, b1))
        a = a1
        b = b1       
    
    return ret

steepest = steepest_descent_f(1, 1, 0.1, n=10)