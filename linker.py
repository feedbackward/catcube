
# linker.py

# All manner of helpers related to computing the function
# which links our particular surrogate's excess risk to the
# excess classification risk incurred.

import numpy as np
import math

import helpers as hlp
import getroots as gtrt

def cube(x, a, b, c, d):
    '''
    Return the value of a cubic polynomial.
    Can vectorize with respect to x or the weights,
    but not both of course.
    '''
    return a*x**3 + b*x**2 + c*x + d

# Coefficients that specify the "plus" version of our single cube condition.
def coef_onecubeplus_A(gam, eta):
    return np.zeros(eta.shape) + 1.0
def coef_onecubeplus_B(gam, eta):
    return np.zeros(eta.shape) + 3*gam
def coef_onecubeplus_C(gam, eta):
    return np.zeros(eta.shape) + 3*gam**2 - 6
def coef_onecubeplus_D(gam, eta):
    alpha = (eta-1)/eta
    return -6 * ( hlp.psi_catnew(math.sqrt(2))/alpha + gam - gam**3/6 )

# Coefficients that specify the "minus" version of our single cube condition.
def coef_onecubeminus_A(gam, eta):
    return np.zeros(eta.shape) + 1.0
def coef_onecubeminus_B(gam, eta):
    return np.zeros(eta.shape) - 3*gam
def coef_onecubeminus_C(gam, eta):
    return np.zeros(eta.shape) + 3*gam**2 - 6
def coef_onecubeminus_D(gam, eta):
    alpha = (eta-1)/eta
    return 6 * ( hlp.psi_catnew(math.sqrt(2))*alpha + gam - gam**3/6 )

def coef_twocube_A(gam, eta):
    alpha = (eta-1)/eta
    return (alpha-1)
def coef_twocube_B(gam, eta):
    alpha = (eta-1)/eta
    return -3*gam*(1+alpha)
def coef_twocube_C(gam, eta):
    alpha = (eta-1)/eta
    return (1-alpha)*(3*gam**2 - 6)
def coef_twocube_D(gam, eta):
    alpha = (eta-1)/eta
    return (1+alpha)*(6*gam-gam**3)


def lossfn(u, gam):
    '''
    The surrogate loss function induced by rho(gamma-u).
    '''
    return hlp.rho_catnew((gam-u))


def condPhiRisk(u, eta, gam):
    return eta * lossfn(u=u, gam=gam) + (1-eta) * lossfn(u=(-u), gam=gam)


def Hfn_single_large(gam, eta):
    deltaval = math.fabs((math.sqrt(2)-gam))
    myA = coef_onecubeminus_A(gam=gam, eta=eta)
    myB = coef_onecubeminus_B(gam=gam, eta=eta)
    myC = coef_onecubeminus_C(gam=gam, eta=eta)
    myD = coef_onecubeminus_D(gam=gam, eta=eta)
    roots = np.zeros(eta.shape)
    for t in range(roots.size):
        etaval = eta[t]
        rootval = gtrt.getroot(a=myA[t], b=myB[t], c=myC[t], d=myD[t])
        rootval = rootval[(np.sign(rootval) == math.copysign(1.0,(etaval-1/2)))]
        rootval = rootval[np.abs(rootval) >= deltaval]
        rootval = rootval[np.abs(rootval) <= gam]
        roots[t] = rootval[0] # should be just one left.
    Hvals = condPhiRisk(u=roots, eta=eta, gam=gam)
    return Hvals


def Hfn_single_small(gam, eta):
    deltaval = math.fabs((math.sqrt(2)-gam))
    myA = coef_onecubeplus_A(gam=gam, eta=eta)
    myB = coef_onecubeplus_B(gam=gam, eta=eta)
    myC = coef_onecubeplus_C(gam=gam, eta=eta)
    myD = coef_onecubeplus_D(gam=gam, eta=eta)
    roots = np.zeros(eta.shape)
    for t in range(roots.size):
        etaval = eta[t]
        rootval = gtrt.getroot(a=myA[t], b=myB[t], c=myC[t], d=myD[t])
        rootval = rootval[(np.sign(rootval) == math.copysign(1.0,(etaval-1/2)))]
        rootval = rootval[np.abs(rootval) >= deltaval]
        rootval = rootval[np.abs(rootval) <= gam]
        roots[t] = rootval[0] # should be just one left.
    Hvals = condPhiRisk(u=roots, eta=eta, gam=gam)
    return Hvals


def Hfn_double(gam, eta):
    myA = coef_twocube_A(gam=gam, eta=eta)
    myB = coef_twocube_B(gam=gam, eta=eta)
    myC = coef_twocube_C(gam=gam, eta=eta)
    myD = coef_twocube_D(gam=gam, eta=eta)
    roots_list = [ gtrt.getroot(a=myA[t], b=myB[t], c=myC[t], d=myD[t])[0] for t in range(myA.size) ]
    roots = np.array(roots_list)
    Hvals = condPhiRisk(u=roots, eta=eta, gam=gam)
    return Hvals


def Hfn(eta, gam):
    
    out = np.zeros(eta.shape)
    
    deltaval = math.fabs((gam-math.sqrt(2)))
    
    idx_zero = np.nonzero(eta == 0)[0]
    idx_one = np.nonzero(eta == 1)[0]
    idx_half = np.nonzero(eta == 1/2)[0]
    idx_small = np.setdiff1d(np.nonzero(eta < 1/2)[0], idx_zero)
    idx_large = np.setdiff1d(np.nonzero(eta > 1/2)[0], idx_one)
    idx_tricky = np.union1d(np.union1d(idx_half, idx_zero), idx_one)
    idx_therest = np.setdiff1d(np.arange(eta.size), idx_tricky)
    eta_large = eta[idx_large]
    eta_small = eta[idx_small]
    eta_therest = eta[idx_therest]
    
    if (gam <= math.sqrt(2)/2):
        # In this case, only solve the double-cube condition.
        Hvals = Hfn_double(gam=gam, eta=eta_therest)
        out[idx_therest] = Hvals
        
    elif (gam > math.sqrt(2)):
        # In this case, only solve the single-cube condition.
        
        # Deal with larger eta values (above 1/2).
        out[idx_large] = Hfn_single_large(gam=gam, eta=eta_large)
        
        # Deal with smaller eta values (below 1/2).
        out[idx_small] = Hfn_single_small(gam=gam, eta=eta_small)
        
    else:
        # Otherwise, a bit more checking is required.
        
        Q = hlp.psi_catnew(deltaval-gam) / hlp.psi_catnew(deltaval+gam)
        tocheck = np.where((eta_therest > 1/2),
                           ((eta_therest-1)/eta_therest),
                           (eta_therest/(eta_therest-1)))
        Hvals = np.zeros(eta_therest.size)
        
        for t in range(eta_therest.size):
            
            etaval = np.take(a=eta_therest, indices=[t])
            
            if (Q < tocheck[t]):
                # Solving the single-cube condition is enough.
                
                if (etaval > 1/2):
                    Hvals[t] = Hfn_single_large(gam=gam, eta=etaval)
                else:
                    Hvals[t] = Hfn_single_small(gam=gam, eta=etaval)
            else:
                # None are relevant; solve the double-cube condition.
                Hvals[t] = Hfn_double(gam=gam, eta=etaval)
            
        out[idx_therest] = Hvals
        
    
    # Deal with the edges and middle.
    out = np.where((eta == 1), 0, out)
    out = np.where((eta == 0), 0, out)
    out = np.where((eta == 1/2), hlp.rho_catnew(gam), out)
        
    return out

    
def linkfn(theta, gam):
    '''
    Domain is theta in [-1,1].
    '''
    return lossfn(u=0, gam=gam) - Hfn(eta=((1+theta)/2), gam=gam)
