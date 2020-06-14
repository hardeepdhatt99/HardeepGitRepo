import math
import numpy as np


def g_abs_err(x, h):
    """Returns the absolute error of computing `g` at `x` if `x` is
    perturbed by a small value `h`.
    """
    p = x + h
    true = (x ** 2) + x - 4
    approx = (p ** 2) + x - 4
    return approx - true

def g_rel_err(x, h):
    """Returns the relative error of computing `g` at `x` if `x` is
    perturbed by a small value `h`.
    """
    true = (x ** 2) + x - 4
    absErr = g_abs_err(x, h)
    return absErr / true


def g_root_abs_err(c, h):
    """Returns the absolute error of finding the (most) positive root of `g` when
    `c` is perturbed by a small value `h`.
    """
    p = c + h
    try:
        approx = ((math.sqrt(1 - (4 * p))) + 1) / 2
        true = ((math.sqrt(1 - (4 * c))) + 1) / 2
        return approx - true
    except ValueError:
        print("Error: The numbers entered resulted in a negative square root")

def g_root_rel_err(c, h):
    """Returns the relative error of finding the (most) positive root of `g` when
    `c` is perturbed by a small value `h`.
    """
    try: 
        absErr = g_root_abs_err(c, h)
        true = ((math.sqrt(1 - (4 * c))) + 1) / 2
        return absErr / true
    except ValueError:
        print("Error: The numbers entered resulted in a negative square root")
    

def f(x):
    return (x - math.sin(x)) / math.pow(x, 3)

def plot_f():
    import matplotlib.pyplot as plt
    xs = [x for x in np.arange(-3.0, 3.0, 0.05) if abs(x) > 0.05]
    ys = [f(x) for x in xs]
    plt.plot(xs, ys, 'bo')
    
# plot_f()

# make a taylor series function that computes sin
def f2(x):
    sin = 0
    fact = 3
    exp = 2
    for r in range(0,100,1):
        term = ((-1) ** r) * (x ** exp) / math.factorial(fact)
        fact += 2
        exp += 2
        sin += term
    return sin / (x ** 2)


# In[ ]:
def approxSin(x):
    return x - ((x**3) / 6)

def fwdError(x):
    return approxSin(x) - math.sin(x)

def backError(x):
    try: 
        return math.asin(approxSin(x)) - x
    except ValueError:
        return "DNE"
    

