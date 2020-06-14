#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np


def newton(f, df, x, n=5):
    """ Return a list of successively better estimate of a root of `f` 
    obtained from applying Newton's method. The argument `df` is the
    derivative of the function `f`. The argument `x` is the initial estimate
    of the root. The length of the returned list is `n + 1`.
    
    Precondition: f is continuous and differentiable
                  df is the derivative of f
    
    >>> def f(x):
    ...     return x * x - 4 * np.sin(x)
    >>> def df(x):
    ...     return 2 * x - 4 * np.cos(x)
    >>> newton(f, df, 3, n=5)
    [3,
     2.1530576920133857,
     1.9540386420058038,
     1.9339715327520701,
     1.933753788557627,
     1.9337537628270216]
    """
    ret = [x]
    for i in range(n):
        if df(x) == 0:
            break
        o = x - (f(x) / df(x))
        ret.append(o)
        x = o
    return ret


def f(x):
    return math.pow(x, 2) - 3 * x + 2
def df(x):
    return 2 * x - 3

newton_root = newton(f, df, 3, 6)


def h1(x):
    return math.pow(x, 3) - 5 * (math.pow(x, 2)) + 8 * x - 4

def dh1(x):
    return 3 * (math.pow(x, 2)) - 10 * x + 8

def h2(x):
    return x * np.cos(20 * x) - x

def dh2(x):
    return np.cos(20 * x) - 20 * x * np.sin(20 * x) - 1

def h3(x):
    return math.pow(np.e, (-2 * x)) + math.pow(np.e, x) - x - 4

def dh3(x):
    return -2 * math.pow(np.e,(-2 * x)) + math.pow(np.e, x) - 1

newton_h1 = newton(h1, dh1, 1.5, 100)
newton_h2 = newton(h2, dh2, 1.5, 100)
newton_h3 = newton(h3, dh3, 1.5, 100)


def secant(f, x0, x1, n=5):
    """ Return a list of successively better estimate of a root of `f` 
    obtained from applying secant method. The arguments `x0` and `x1` are
    the two starting guesses. The length of the returned list is `n + 2`.
    
    >>> secant(lambda x: x ** 2 + x - 4, 3, 2, n=6)
    [3,
     2,
     1.6666666666666667,
     1.5714285714285714,
     1.5617977528089888,
     1.5615533980582523,
     1.561552812843596,
     1.5615528128088303]
    """
    ret = [x0, x1]
    xk = x1
    xkSub1 = x0
    for i in range(n):
        if (f(xk) - f(xkSub1)) == 0:
            break
        xkPlus1 = xk - f(xk) * ((xk - xkSub1) / (f(xk) - f(xkSub1)))
        ret.append(xkPlus1)
        xkSub1 = xk
        xk = xkPlus1        
    return ret


def f(x):
    return math.pow(x,3) + math.pow(x,2) + x - 4
secant_root = secant(f, 3, 2, 19)