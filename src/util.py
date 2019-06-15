import numpy as np
import numpy.polynomial as npp


def GaussHermite(n):
    '''
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.hermite.hermgauss.html

    Gauss-Hermite quadrature.

    Computes the sample points and weights for Gauss-Hermite quadrature. 
    These sample points and weights will correctly integrate 
    polynomials of degree 2*deg - 1 or less over the interval 
    [-inf, inf] with the weight function f(x) = exp(-x^2).
    '''
    return npp.hermite.hermgauss(n)
