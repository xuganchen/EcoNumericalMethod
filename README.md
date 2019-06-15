# EcoNumericalMethod
## Introduction

The code for **Numerical Method in Economy** in Zhejiang University (Instructor: Prof. Eric R. Young).

The source code in the *./src/* fodler, and *./figure/* include the figures coming from the code.

Of course, all of this code is based on the *MATLAB* code given by the Prof. Eric R. Young and is translated into *Python* code by me, as my PC is inconvenient to use *MATLAB* (oh, *MATLAB* is too heavy...) and I am more familiar with *Python*.

## Code

* zje_projection.py: using projection to solve growth model
* zje_projection_2.py: using projection to solve growth model
* zje_projection_pchip.py: using projection to solve growth model
* zje_dp_pchip.py: dynamic programming with pchip
* zje_dp_pchip2.py: dynamic programming with pchip
* zje_dp_discrete.py: discrete dynamic programming
* growth_model_zje_run.py: using first-, second-  and third- order solutions to solve growth model (Now, I only finish the first-order solution, and I will finish the remain part in the future.)
* LQ1.py: compute deterministic LQ problem and compute Euler equation errors
<<<<<<< HEAD

## More details

As we know, *MATLAB* is quite suitable for the matrix operations, symbolic operations and optimization. So, when I use *Python*, first I have to find some simple but powerful packages to do these operations. Of course, *Python* does have these functions.

* For matrix operations, I use [**NumPy**](https://www.numpy.org). 

  > NumPy is the fundamental package for scientific computing with Python. It contains among other things:
  >
  > * a powerful N-dimensional array object
  >
  > - sophisticated (broadcasting) functions
  > - tools for integrating C/C++ and Fortran code
  > - useful linear algebra, Fourier transform, and random number capabilities

* For symbolic operations, [**SymPy**](https://www.sympy.org/en/index.html) is quite powerful.

  > SymPy is a Python library for symbolic mathematics. It aims to become a full-featured computer algebra system (CAS) while keeping the code as simple as possible in order to be comprehensible and easily extensible. SymPy is written entirely in Python.

* For optimization, [**SciPy**](https://www.scipy.org) is my first choice.

  > … including signal processing, optimization, statistics and much more.
