# -*- coding: utf-8 -*-
from math import isinf
import numpy as np

def DictParts(m, n):
    D = []
    Last = np.array([[0], [m], [m]])
    end = 0
    for i in range(n):
        NewLast = np.empty((3, 0), dtype=np.int32)
        ncol = Last.shape[1]
        for j in range(ncol):
            record = Last[:, j]
            lack = record[1]
            l = min(lack, record[2])
            if l > 0 :
                D.append((record[0]+1, end+1))
                x = np.empty((3, l), dtype=np.int32)
                for k in range(l):
                    x[:, k] = np.array([end+k+1, lack-k-1, k+1])
                NewLast = np.hstack((NewLast, x))
                end += l
        Last = NewLast
    return (dict(D), end)
   
def _N(dico, kappa):
    kappa = kappa[kappa > 0]
    l = len(kappa)
    if l == 0:
        return 1
    return dico[_N(dico, kappa[:(l-1)])] + kappa[-1] 

def _T(alpha, a, b, kappa):
    i = len(kappa)
    if i == 0 or kappa[0] == 0:
        return 1
    c = kappa[-1] - 1 - (i - 1) / alpha
    d = alpha * kappa[-1] - i
    s = np.asarray(range(1, kappa[-1]), dtype=np.int32)
    e = d - alpha * s + np.array(
        list(map(lambda j: np.count_nonzero(kappa >= j), s))
    )
    g = e + 1
    ss = range(i-1)
    f = alpha * kappa[ss] - (np.asarray(ss) + 1) - d
    h = alpha + f
    l = h * f
    prod1 = np.prod(a + c)
    prod2 = np.prod(b + c)
    prod3 = np.prod((g - alpha) * e / (g * (e + alpha)))
    prod4 = np.prod((l - f) / (l + h))
    out = prod1 / prod2 * prod3 * prod4
    return 0 if isinf(out) or np.isnan(out) else out


def betaratio(kappa, mu, k, alpha):
    muk = mu[k-1]
    t = k - alpha * muk
    prod1 = prod2 = prod3 = 1
    if k > 0:
        u = np.array(
            list(map(lambda i: t + 1 - i + alpha * kappa[i-1], range(1,k+1)))
        )
        prod1 = np.prod(u / (u + alpha - 1))
    if k > 1:
        v = np.array(
            list(map(lambda i: t - i + alpha * mu[i-1], range(1, k)))
        )
        prod2 = np.prod((v + alpha) / v)
    if muk > 1:
        muPrime = dualPartition(mu)
        w = np.array(
            list(map(lambda i: muPrime[i-1] - t - alpha * i, range(1, muk)))
        )
        prod3 = np.prod((w + alpha) / w)
    return alpha * prod1 * prod2 * prod3

def dualPartition(kappa):
    out = []
    if len(kappa) > 0 and kappa[0] > 0:
        for i in range(1, kappa[0]+1):
            out.append(np.count_nonzero(kappa >= i))
    return np.asarray(out, dtype=np.int32)


def hypergeomI(m, alpha, a, b, n, x):
    def summation(i, z, j, kappa):
        def go(kappai, zz, s):
            if i == 0 and kappai > j or i > 0 and kappai > min(kappa[i-1], j):
                return s
            kappap = np.vstack((kappa, [kappai]))
            t = _T(alpha, a, b, kappap[kappap > 0])
            zp = zz * x * (n - i + alpha * (kappai -1)) * t
            sp = s
            if j > kappai and i <= n:
                sp += summation(i+1, zp, j - kappai, kappap)
            spp = sp + zp
            return go(kappai + 1, zp, spp)
        return go(1, z, 0)
    return 1 + summation(0, 1, m, np.empty((0,1), dtype=np.int32))

def is_square_matrix(x):
    x = np.asarray(x)
    return x.ndim == 2 and x.shape[0] == x.shape[1]

def hypergeomPQ(m, a, b, x, alpha=2):
    """Hypergeometric function of a matrix argument.
    
    :param m: truncation weight of the summation, a positive integer
    :param a: the "upper" parameters, a numeric or complex vector, possibly empty (or `None`)
    :param b: the "lower" parameters, a numeric or complex vector, possibly empty (or `None`)
    :param x: a numeric or complex vector, the eigenvalues of the matrix
    :param alpha: the alpha parameter, a positive number
    
    """
    if a is None or len(a) == 0:
        a = np.array([])
    else:
        a = np.asarray(a)
    if b is None or len(b) == 0:
        b = np.array([])
    else:
        b = np.asarray(b)
    x = np.asarray(x, dtype=np.int32)
    n = len(x)
    if all(x == x[0]):
        return hypergeomI(m, alpha, a, b, n, x[0])
    def jack(k, beta, c, t, mu, jarray, kappa, nkappa):
        lmu = len(mu)
        for i in range(max(1, k), (np.count_nonzero(mu)+1)):
            u = mu[i-1]
            if lmu == i or u > mu[i]:
                gamma = beta * betaratio(kappa, mu, i, alpha)
                mup = mu.copy()
                mup[i-1] = u - 1
                mup = mup[mup > 0]
                if len(mup) >= i and u > 1:
                    jack(i, gamma, c + 1, t, mup, jarray, kappa, nkappa)
                else:
                    if nkappa > 1:
                        if len(mup) > 0:
                            jarray[nkappa-1, t-1] += (
                                gamma * jarray[_N(dico, mup)-2, t-2] 
                                * x[t-1]**(c+1)
                            )
                        else:
                            jarray[nkappa-1, t-1] += gamma * x[t-1]**(c+1)
        if k == 0:
            if nkappa > 1:
                jarray[nkappa-1, t-1] += jarray[nkappa-1, t-2]
        else:
            jarray[nkappa-1, t-1] += (
                beta * x[t-1]**c * jarray[_N(dico, mu)-2, t-2]
            )
    def summation(i, z, j, kappa, jarray):
        def go(kappai, zp, s):
            if (
                    i == n or i == 0 and kappai > j 
                    or i > 0 and kappai > min(kappa[-1], j)
                ):
                return s
            kappap = np.concatenate((kappa, [kappai]))
            nkappa = _N(dico, kappap) - 1
            zpp = zp * _T(alpha, a, b, kappap)
            if nkappa > 1 and (len(kappap) == 1 or kappap[1] == 0):
                 jarray[nkappa-1, 0] = (
                     x[0] * (1 + alpha * (kappap[0] - 1)) * jarray[nkappa-2, 0]
                 )
            for t in range(2, n+1):
                jack(0, 1.0, 0, t, kappap, jarray, kappap, nkappa)
            sp = s + zpp * jarray[nkappa-1, n-1]
            if j > kappai and i <= n:
                spp = summation(i+1, zpp, j-kappai, kappap, jarray)
                return go(kappai+1, zpp, sp + spp)
            return go(kappai+1, zpp, sp)
        return go(1, z, 0)
    (dico, Pmn) = DictParts(m, n)
    T = type(x[0])
    print(T)
    J = np.zeros((Pmn, n), dtype=T)
    J[0, :] = np.cumsum(x)
    return 1 + summation(0, T(1), m, np.empty(0, dtype=np.int32), J)  # Changed by me

###############################################################################
# multivariate Gamma ##########################################################
###############################################################################
import numpy as np
from scipy.special import loggamma, gammaln
from math import log, pi, floor
import math
import cmath

def _lmvgamma(x, p):
    C = p * (p-1) / 4 * log(pi)
    if isinstance(x, complex):
        S = np.sum(loggamma(x.real + np.arange(1-p, 1)/2 + 1j * np.repeat(x.imag, p)))
    else:
        S = np.sum(gammaln(x + np.arange(1-p, 1)/2))
    return C + S

def lmvgamma(x, p):
    """Log multivariate Gamma function.
    
    :param x: a real or complex number with a positive real part
    :param p: a positive integer
    
    """
    if not isinstance(p, int) or p < 1:
        raise ValueError("`p` must be a positive integer.")
    if x.real < 0:
        raise ValueError("`x` can be a complex number but only with a positive real part.")
    return _lmvgamma(x, p)

def pochhammer(z, n):
    return np.prod(z + np.arange(0, n))

def mvgamma(x, p):
    """Multivariate Gamma function.
    
    :param x: a real or complex number but not a negative integer
    :param p: a positive integer
    
    """
    if not isinstance(p, int) or p < 1:
        raise ValueError("`p` must be a positive integer.")
    if x.imag == 0 and x.real < 0 and floor(x) == x:
        raise ValueError("`x` must not be a negative integer.")
    if x.real > 0:
        out = math.exp(_lmvgamma(x, p))
    else:
        n = floor(-x.real) + 1
        out = cmath.exp(_lmvgamma(x+n, p)) / pochhammer(x, n)
    return out
        

###############################################################################
# tests #######################################################################
###############################################################################
from scipy.linalg import toeplitz
import math
import cmath
from jackpy.jack import SchurPol, JackPol, ZonalPol, ZonalQPol
from sympy.combinatorics.partitions import IntegerPartition
"""
# 0F0 is the exponential of the trace
X = toeplitz([3, 2, 1]) / 10
x = np.linalg.eigvals(X)
obtained = hypergeomPQ(10, [], [], x)
expected = math.exp(np.sum(np.diag(X)))
print(math.isclose(obtained, expected, rel_tol=1e-6))
X = toeplitz([3j, 2, 1]) / 10
x = np.linalg.eigvals(X)
obtained = hypergeomPQ(10, [], [], x)
expected = cmath.exp(np.sum(np.diag(X)))
print(cmath.isclose(obtained, expected, rel_tol=1e-6))

# 1F0 is det(I-X)^(-a)
X = toeplitz([3, 2, 1]) / 100
x = np.linalg.eigvals(X)
obtained = hypergeomPQ(15, [3], [], x)
expected = np.linalg.det(np.eye(3)-X)**(-3)
print(math.isclose(obtained, expected, rel_tol=1e-6))
X = toeplitz([2, 2, 1]) / 100
x = np.linalg.eigvals(X)
obtained = hypergeomPQ(15, [4j], [], x)
expected = np.linalg.det(np.eye(3)-X)**(-4j)
print(cmath.isclose(obtained, expected, rel_tol=1e-6))
X = toeplitz([3j, 2, 1]) / 100
x = np.linalg.eigvals(X)
obtained = hypergeomPQ(15, [3], [], x)
expected = np.linalg.det(np.eye(3)-X)**(-3)
print(cmath.isclose(obtained, expected, rel_tol=1e-6))
X = toeplitz([2j, 2, 1]) / 100
x = np.linalg.eigvals(X)
obtained = hypergeomPQ(15, [4j], [], x)
expected = np.linalg.det(np.eye(3)-X)**(-4j)
print(cmath.isclose(obtained, expected, rel_tol=1e-6))

# Some values for 2F1
obtained = hypergeomPQ(10, [1, 2], [3], [0.2, 0.5])
print(math.isclose(obtained, 1.79412894456143, rel_tol=1e-6))
obtained = hypergeomPQ(10, [1j, 2], [3j], [0.2, 0.5])
print(cmath.isclose(obtained, 1.677558924-0.183004016j, rel_tol=1e-6))
obtained = hypergeomPQ(10, [1, 2], [3], [0.2j, 0.5])
print(cmath.isclose(obtained, 1.513810425+0.20576184j, rel_tol=1e-6))
obtained = hypergeomPQ(10, [1, 2j], [3], [0.2j, 0.5])
print(cmath.isclose(obtained, 0.7733140719+0.3092059749j, rel_tol=1e-6))

# Gauss formula for 2F1
a = 1
b = 2
c = 9
o1 = mvgamma(c,3)*mvgamma(c-a-b,3)/mvgamma(c-a,3)/mvgamma(c-b,3)
o2 = hypergeomPQ(100, [a, b], [c], [1,1,1])
print(math.isclose(o1, o2, rel_tol=1e-6))
a = 1j
o1 = mvgamma(c,3)*mvgamma(c-a-b,3)/mvgamma(c-a,3)/mvgamma(c-b,3)
o2 = hypergeomPQ(100, [a, b], [c], [1,1,1])
print(cmath.isclose(o1, o2, rel_tol=1e-6))

# Herz's relation for 2F1
X = toeplitz([3j,2,1])/100
x = np.linalg.eigvals(X)
o1 = hypergeomPQ(15, [1,2j], [3], x)
x = np.linalg.eigvals(-X @ np.linalg.inv(np.eye(3)-X))
o2 = np.linalg.det(np.eye(3)-X)**(-2j) * hypergeomPQ(15, [3-1,2j], [3], x)
print(cmath.isclose(o1, o2, rel_tol=1e-6))
"""

# Expansion in Jack Polynomials
def genpoch(a, kappa, alpha):
    return np.prod([np.prod(a - i/alpha + np.arange(kappa[i])) for i in range(len(kappa))])
def coeff0(a, b, alpha):
    return (
        genpoch(a[0], [0], alpha) * genpoch(a[1], [0], alpha) 
        / genpoch(b[0], [0], alpha) / genpoch(b[1], [0], alpha)        
    )
def coeff1(a, b, alpha):
    return (
        genpoch(a[0], [1], alpha) * genpoch(a[1], [1], alpha) 
        / genpoch(b[0], [1], alpha) / genpoch(b[1], [1], alpha)        
    )
def coeff2(a, b, alpha):
    return (
        genpoch(a[0], [2], alpha) * genpoch(a[1], [2], alpha) 
        / genpoch(b[0], [2], alpha) / genpoch(b[1], [2], alpha) / 2     
    )
def coeff11(a, b, alpha):
    return (
        genpoch(a[0], [1,1], alpha) * genpoch(a[1], [1,1], alpha) 
        / genpoch(b[0], [1,1], alpha) / genpoch(b[1], [1,1], alpha) / 2     
    )


"""
alpha = 1
a = [2, 3]
b = [4, 1j]
x = [0.3j, 0.7]
  
schur = (
    coeff0(a, b, alpha) * SchurPol(2, IntegerPartition(np.asarray([],dtype=int)))
    + coeff1(a, b, alpha) * SchurPol(2, IntegerPartition([1]))
    + coeff11(a, b, alpha) * SchurPol(2, IntegerPartition([1,1]))
    + coeff2(a, b, alpha) * SchurPol(2, IntegerPartition([2]))
)
expected = complex(*schur.eval(x).as_real_imag())
obtained = hypergeomPQ(2, a, b, x, alpha)
cmath.isclose(expected,obtained)
"""