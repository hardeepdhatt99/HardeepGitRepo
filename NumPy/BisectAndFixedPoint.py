#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np

def bisect(f, a, b, n):
    """Returns a list of length n+1 of intervals 
    where f(x) = 0 lies, where each interval is half 
    the size of the previous, and is obtained using
    the interval bisection method.
    
    Precondition: f continuous,
                  a < b
                  f(a) and f(b) have opposite signs
                  
    Example:
    >>> bisect(lambda x: x - 1, -0.5, 2, n=5)
    [(-0.5, 2),
     (0.75, 2),
     (0.75, 1.375),
     (0.75, 1.0625),
     (0.90625, 1.0625),
     (0.984375, 1.0625)]
    """
    ret = []
    ret.append((a, b))
    for i in range(n):
        m = a + ((b - a)/2)
        if (np.sign(f(m)) == np.sign(f(a))):
            a = m
            ret.append((a, b))
        else:
            b = m
            ret.append((a, b))
    return ret        


def fixed_point(f, x, n=20):
    """ Return a list lst = [x, f(x), f(f(x)), ...] with 
    `lst[i+1] = f(lst[i])` and `len(lst) == n + 1`
    
    >>> fixed_point(lambda x: math.sqrt(x + 1), 3, n=5)
    [3,
     2.0,
     1.7320508075688772,
     1.6528916502810695,
     1.6287699807772333,
     1.621348198499395]
    """
    ret = []
    ret.append(x)
    for i in range(n):
        ret.append(f(x))
        x = f(x)
    return ret    

g1 = fixed_point(lambda x: (x**2 + 2) / 3, 3, 12)
g2 = fixed_point(lambda x: math.sqrt(3*x - 2), 3)
g3 = fixed_point(lambda x: 3 - 2/x, 3)