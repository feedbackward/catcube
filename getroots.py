
# getroots.py

import math
import numpy as np


# Functions which help us with root-finding operations.

def discrim(a, b, c, d):
    '''
    Discriminant of a general cubic polynomial.
    '''
    return 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2

def del_0(a, b, c, d):
    '''
    Computational term, Delta_0.
    '''
    return b**2 - 3*a*c

def del_1(a, b, c, d):
    '''
    Computational term, Delta_1.
    '''
    return 2*b**3 - 9*a*b*c + 27*a**2*d

def getroot(a, b, c, d):
    '''
    Root-computing function.
    Assumes that a, b, c, d are scalars.
    Returns a list of one or more roots.
    '''
    discrimval = discrim(a=a, b=b, c=c, d=d)
    del0 = del_0(a=a, b=b, c=c, d=d)
    del1 = del_1(a=a, b=b, c=c, d=d)
    
    if discrimval < 0:

        # In this case, there is only one real root.
        
        num = del1 + math.sqrt((del1**2 - 4*del0**3))
        if (num >= 0):
            C_val = (num/2)**(1/3)
        else:
            C_val = -(math.fabs(num)/2)**(1/3)
        root = -(b + C_val + del0/C_val) / (3*a)
        return np.array([root]) # a single real value.
    
    elif discrimval == 0:

        # In this case, there exist multiple roots, and they are
        # all real.
        
        if (del0 == 0):
            # If Delta_0 is zero, then just one "triple" root.
            root = -b/(3*a)
            return np.array([root]) # a single real value.
        
        else:
            # If Delta_0 is non-zero, then we have two distinct
            # roots, one "double" root and one "single" root.
            root_double = (9*a*d - b*c) / (2*del0)
            root_simple = (4*a*b*c - 9*a**2*d - b**3) / (a*del0)
            return np.array([root_double, root_simple])
        
    else:

        # In this case, all roots are distinct and real, and in
        # general these need not be *rational*. If not rational,
        # this is the "casus irreducibilis", and the roots cannot
        # be expressed using real "radicals".

        # In this case, a very convenient strategy is to use the
        # trigonometric functions, which are easily computed by
        # machines and offer "direct" access to the roots.
        
        p = (3*a*c - b**2) / (3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)

        # Since p < 0 follows from Delta > 0, we can use the
        # sleek trigonometric approach.
        k = math.sqrt(-p/3)
        e = -q / (2*k**3)
        ang_base = math.acos(e) / 3
        all_angles = [ang_base,
                      (2*math.pi/3-ang_base),
                      (2*math.pi/3+ang_base)]
        all_z = [ math.cos(ang) for ang in all_angles ]
        all_v = [ 2*k*z for z in all_z ]
        roots = [ v-b/(3*a) for v in all_v ]
        return np.array(roots)

