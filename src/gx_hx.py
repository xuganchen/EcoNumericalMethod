import numpy as np
import numpy.linalg as nplg
import scipy.linalg as splg


def qzswitch(i, A, B, Q, Z):
    '''
    %function [A,B,Q,Z] = qzswitch(i,A,B,Q,Z)
    % Written by Chris Sims
    % Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges
    % diagonal elements i and i+1 of both A and B, while maintaining 
    % Q'AZ' and Q'BZ' unchanged.  Does nothing if ratios of diagonal elements
    % in A and B at i and i+1 are the same.  Aborts if diagonal elements of
    % both A and B are zero at either position.
    %
    '''

    a = A[i - 1, i - 1]
    d = B[i - 1, i - 1]
    b = A[i - 1, i]
    e = B[i - 1, i]
    c = A[i + 1, i + 1]
    f = B[i + 1, i + 1]

    wz = np.array([c * e - f * b, c * d - f * a]).reshape(-1, 1)
    xy = np.array([b * d - e * a, c * d - f * a]).reshape(-1, 1)

    n = np.sqrt(wz * wz.T)
    m = np.sqrt(xy * xy.T)

    if n == 0:
        return
    else:
        wz = np.dot(nplg.inv(n), wz)
        xy = np.dot(nplg.inv(m), xy)
        wz = np.concatenate(
            [wz, np.concatenate([-wz[1], wz[0]]).reshape(-1, 1)], axis=1)
        xy = np.concatenate(
            [xy, np.concatenate([-xy[1], xy[0]]).reshape(-1, 1)], axis=1)

        A[i - 1:i + 1, :] = np.dot(xy, A[i - 1:i + 1, :])
        B[i - 1:i + 1, :] = np.dot(xy, B[i - 1:i + 1, :])
        Q[i - 1:i + 1, :] = np.dot(xy, Q[i - 1:i + 1, :])
        A[:, i - 1:i + 1] = np.dot(A[:, i - 1:i + 1], wz)
        B[:, i - 1:i + 1] = np.dot(B[:, i - 1:i + 1], wz)
        Z[:, i - 1:i + 1] = np.dot(Z[:, i - 1:i + 1], wz)


def qzdiv(stake, A, B, Q, Z):
    '''
    % Written by Chris Sims
    %
    % Takes U.T. matrices A, B, orthonormal matrices Q,Z, rearranges them
    % so that all cases of abs(B(i,i)/A(i,i))>stake are in lower right 
    % corner, while preserving U.T. and orthonormal properties and Q'AZ' and
    % Q'BZ'.
    '''
    n, jnk = A.shape
    root = np.abs(np.array([np.diag(A), np.diag(B)]).T)
    root[:, 0] = root[:, 0] - (root[:, 0] < 1.e-13) * (root[:, 0] + root[:, 1])
    root[:, 1] = root[:, 1] / root[:, 0]
    for i in range(n, 0, -1):
        m = -1
        for j in range(i, 0, -1):
            if (root[j - 1, 1] > stake) or (root[j - 1, 1] < -0.1):
                m = j
                break
        if m == -1:
            break
        for k in range(m, i):
            A, B, Q, Z = qzswitch(k, A, B, Q, Z)
            tmp = root[k - 1, 2]
            root[k - 1, 2] = root[k, 1]
            root[k, 1] = temp

    return A, B, Q, Z


def solab(a, b, nk):
    '''
    % Function: solab
    % Purpose: Solves for the recursive representation of the stable solution to a system
    % of linear difference equations.
    % Inputs: Two square matrices a and b and a natural number nk
    % a and b are the coefficient matrices of the difference equation
    % a*x(t+1) = b*x(t)
    % where x(t) is arranged so that the state variables come first, and
    % nk is the number of state variables.
    % Outputs: the decision rule f and the law of motion p. If we write
    % x(t) = [k(t);u(t)] where k(t) contains precisely the state variables, then
    % u(t)   = f*k(t) and
    % k(t+1) = p*k(t).
    % Calls: qzdiv
    '''
    s, t, q, z = splg.qz(a, b)
    s, t, q, z = qzdiv(1, s, t, q, z)

    z21 = z[nk + 1:, 1:nk + 1]
    z11 = z[1: nk + 1, 1: nk + 1]

    z11i = np.dot(nplg.inv(z11), np.eye(nk))
    s11 = s[1: nk + 1, 1: nk + 1]
    t11 = t[1: nk + 1, 1: nk + 1]

    dyn = np.dot(nplg.inv(s11), t11)
    f = z21 * z11i
    p = z11 * dyn * z11i
    return f, p


def gx_hx(config):
    '''
    %This program computes the matrices gx and hx that define the first-order approximation 
    %of the DSGE model. That is, if 
    %E_t[f(yp,y,xp,x)=0, then the solution is of the form
    %xp = h(x,sigma) + sigma * eta * ep
    %y = g(x,sigma).
    %The first-order approximations to the functions g and h around the point (x,sigma)=(xbar,0), where xbar=h(xbar,0), are:
    %h(x,sigma) = xbar + hx (x-xbar) 
    %and
    %g(x,sigma) = ybar + gx * (x-xbar),
    %where ybar=g(xbar,0). 
    %Inputs: fy fyp fx fxp
    %Outputs: gx hx
    %Calls solab.m (by Paul Klein)
    %(c) Stephanie Schmitt-Grohe and Martin Uribe
    %Date July 17, 2001
    '''
    A = np.concatenate([-config['nfxp'], -config['nfyp']], axis=1)
    B = np.concatenate([-config['nfx'], -config['nfy']], axis=1)
    nk = config['nfx'].shape[1]
    ngx, nhx = solab(A, B, nk)
    config['ngx'] = ngx
    config['nhx'] = nhx
    return config
