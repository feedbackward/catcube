
# helpers.py

import math
import numpy as np


### Influence functions and their parents. ###

def rho_catnew(u):
    '''
    Parent derived from the influence function of
    Catoni and Giulini (2017)
    '''
    return np.where((np.abs(u) > math.sqrt(2)),
                    (np.abs(u)*2*math.sqrt(2)/3-0.5),
                    (u**2/2-u**4/24))

def psi_catnew(u):
    '''
    Influence function of Catoni and Giulini (2017).
    '''
    return np.where((np.abs(u) > math.sqrt(2)),
                    (np.sign(u)*2*math.sqrt(2)/3),
                    (u-u**3/6))

CONST_catnew = math.sqrt(81/32)


def rho_lcosh(u):
    '''
    A classic example of a Catoni-type function (parent).
    '''
    return np.log(np.cosh(u))

def psi_lcosh(u):
    '''
    A classic example of a Cat-type influence function.
    '''
    return np.tanh(u)


def rho_hinge(u):
    '''
    The hinge function used to enforce relaxed constraints
    in a margin-maximization setting.
    '''
    return np.where((u>=0), u, 0)

def psi_hinge(u):
    '''
    The first-order derivative of the hinge function used to
    enforce relaxed constraints in a margin-maximization
    setting.
    '''
    return np.where((u>=0), 1, 0)



    


    
