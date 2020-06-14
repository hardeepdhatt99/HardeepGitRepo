import math
import numpy as np

# These are the constants we'll use in our code
BASE = 3
PRECISION = 5
L = -3
U = +3

# We will represent a floating number as a tuple (mantissa, exponent, sign)
# With: mantissa -- a list of integers of length PRECISION
#       exponent -- an integer between L and U (inclusive)
#       sign     -- either 1 or -1
example_float = ([1, 0, 0, 1, 0], 1, 1)

total_num_floats = 2 * (BASE - 1) * (BASE ** (PRECISION - 1)) * (U - L + 1) + 1


def is_valid_float(float_value):
    """Returns a boolean representing whether the float_value is a valid,
    normalized float in our floating point system.

    >>> is_valid_float(example_float)
    True
    """

    (mantissa, exponent, sign) = float_value
    s = (sign == -1) or (sign == 1)
    e = (exponent >= L) and (exponent <= U)
    l = len(mantissa) == PRECISION
    for x in mantissa:
        if (x >= 0) and (x <= 2):
            m = True 
        else:
            m = False
            break
    return s and m and e and l



def to_num(float_value):
    """Return a Python floating point representation of `float_val`
    These examples are for your understanding, and your actual output
    might vary slightly.
    
    >>> float_val(example_float)
    3.111111111111111
    """
    (mantissa, exponent, sign) = float_value
    exp = 0
    m = 0
    for x in mantissa:
        m += x / (BASE ** exp)
        exp += 1
    return m * (BASE ** exponent) * sign


def add_float(float1, float2):
    """Return a floating-point representation of the form (mantissa, exponent, sign)
    that is the sum of `float1` and `float2`.
    
    >>> add_float(example_float, example_float)
    ([1, 0, 0, 1, 0], 2, 1])
    """
    (mantissa1, exponent1, sign1) = float1
    (mantissa2, exponent2, sign2) = float2

    # You may assume that sign1 and sign2 are positive
    assert (sign1 > 0) and (sign2 > 0)
    
    if (to_num(float1) + to_num(float2)) > to_num(([2, 2, 2, 2, 2], 3, 1)):
        raise ValueError
    
    if exponent1 > exponent2:
        diff = int(math.fabs(exponent1) + math.fabs(exponent2))
        finalExp = exponent1
        for x in range (0,diff,1):
            mantissa1.append(0)
            mantissa2.insert(0,0)
    elif exponent2 > exponent1:
        finalExp = exponent2
        diff = int(math.fabs(exponent1) + math.fabs(exponent2))
        for x in range (0,diff,1):
            mantissa2.append(0)
            mantissa1.insert(0,0)
    else:
        finalExp = exponent1
    
    newMantissa = []
    for x in mantissa1:
        newMantissa.append(0)
    
    for i in range(len(mantissa1) - 1, -1, -1):        
        if (mantissa1[i] + mantissa2[i] + newMantissa[i]) == 4:           
            newMantissa[i-1] = 1
            newMantissa[i] = 1            
        elif (mantissa1[i] + mantissa2[i] + newMantissa[i]) == 3:  
            newMantissa[i-1] = 1
            newMantissa[i] = 0
        elif (mantissa1[i] + mantissa2[i] + newMantissa[i]) < 3: 
            newMantissa[i] = newMantissa[i] + mantissa1[i] + mantissa2[i]
        else: 
            raise ValueError 
                
    finalMan = newMantissa[:5]
                
    return (finalMan, finalExp, 1)

def h1(x, n):
    """Returns a list of the first n terms of the Taylor Series expansion of 1/(1-x)."""
    return [pow(x,i) for i in range(n)]

def h2(x, n):
    """Returns a list of the first n terms of the Taylor Series expansion of e^x."""
    return [pow(x,i)/math.factorial(i) for i in range(n)]


ns = [20, 40, 60, 80, 100, 120, 140, 160]
exp_estimates = []
for x in ns:
    exp_estimates.append(sum(h2(-30, x)))

def z(n):
    a = pow(2.0, n) + 10.0
    b = (pow(2.0, n) + 5.0) + 5.0
    return a - b

nonzero_zn = []

for x in range(0,1023,1):
    if z(x) != 0:
        nonzero_zn.append(x)