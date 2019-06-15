import os
import numpy as np
import numpy.linalg as nplg
import matplotlib.pyplot as plt


def uLQ1(x, p):
    const = x[0]
    k = x[1]
    kp = x[2]
    J = p[0]
    alpha = p[1]
    beta = p[2]
    delta = p[3]
    c = J * np.power(k, alpha) + (1.0 - delta) * k - kp
#     print(c)
    u = np.log(c)
    return u


def LQ1(ns, nz, nd,
        J, alpha, beta, delta,
        m_kgrid2,
        iternum,
        figurepath=None):
    '''
    %program to compute deterministic LQ problem and compute Euler equation
    %errors
    '''
    nst = nz + ns
    nvec = nz + ns + nd

    kbar = np.power(alpha * beta * J / (1.0 - beta *
                                        (1.0 - delta)), 1.0 / (1.0 - alpha))
    vec = np.array([1.0, kbar, kbar])
    param = np.array([J, alpha, beta, delta])

    # %linear approximation
    Du = np.zeros(nvec)
    for i in range(nvec):
        h = np.zeros(nvec)
        h[i] = 1.0e-06 * vec[i]
        if h[i] == 0:
            h[i] = 1.0e-06
        Du[i] = 1 / h[i] / 2 * (uLQ1(vec + h, param) - uLQ1(vec - h, param))

    # %quadratic approximation
    D2u = np.zeros((nvec, nvec))
    for i in range(nvec):
        for j in range(nvec):
            h1 = np.zeros(nvec)
            h2 = np.zeros(nvec)
            h1[i] = 1.0e-03 * vec[i]
            h2[j] = 1.0e-03 * vec[j]
            if h1[i] == 0:
                h1[i] = 1.0e-03
            if h2[j] == 0:
                h2[j] = 1.0e-03
            D2u[i, j] = 1 / 4 / h1[i] / h2[j] * (uLQ1(vec + h1 + h2, param) - uLQ1(
                vec - h1 + h2, param) - uLQ1(vec + h1 - h2, param) + uLQ1(vec - h1 - h2, param))
    D2u = 0.5 * D2u

    R = D2u[nst: nvec, nst: nvec]
    T = D2u[nst: nvec, 1: nst]
    S = D2u[1: nst, 1: nst]

    lk = 0.5 * (Du[nst:nvec].T - 2 * vec[1:nst].T *
                T.T - 2 * vec[nst:nvec].T * R).T
    W = np.concatenate([lk, T], axis=1)

    ln = 0.5 * (Du[1:nst].T - 2 * vec[1:nst].T * S - 2 * vec[nst:nvec].T * T).T
    aa = np.array(uLQ1(vec, param) + np.dot(np.dot(vec[1:nvec].T, D2u[1:nvec, 1:nvec]),
                                            vec[1:nvec]) - np.dot(Du[1:nvec].T, vec[1:nvec])).reshape(1, -1)
    O = np.concatenate([np.concatenate([aa, ln.T], axis=1),
                        np.concatenate([ln, S], axis=1)])
    Q = np.concatenate([np.concatenate([O, W.T], axis=1),
                        np.concatenate([W, R], axis=1)])

    # %partitioning Q
    Q11 = Q[0:nz, 0:nz]
    Q12 = Q[0:nz, nz:nst]
    Q13 = Q[0:nz, nst:nvec]
    Q21 = Q12.T
    Q22 = Q[nz:nst, nz:nst]
    Q23 = Q[nz:nst, nst:nvec]
    Q31 = Q13.T
    Q32 = Q23.T
    Q33 = Q[nst:nvec, nst:nvec]

    # % defining constraint matrices
    A = np.array([1.0])
    B1 = np.array([0.0])
    B2 = np.array([0.0])
    B3 = np.array([1.0])

    # %iteration on matrix Riccatti equation
    P = np.zeros((nz + ns, nz + ns))
    P11 = P[0:nz, 0:nz]
    P12 = P[0:nz, nz:nz + ns]
    P21 = P12.T
    P22 = P[nz:nz + ns, nz:nz + ns]
    dF = - (Q33 + beta * np.dot(np.dot(B3.T, P22), B3))
    F1 = (Q31 + beta * np.dot(np.dot(B3.T, P21), A) +
          beta * np.dot(np.dot(B3.T, P22), B1)) / dF
    F2 = (Q32 + beta * np.dot(np.dot(B3.T, P22), B2)) / dF
    F = np.concatenate([F1, F2], axis=1)
    AF = np.concatenate([np.concatenate([A.reshape(1, -1), np.zeros((nz, ns))], axis=1),
                         np.concatenate([B1 + B3 * F1, B2 + B3 * F2], axis=1)])
    P_1 = np.concatenate([np.concatenate([Q11, Q12], axis=1),
                          np.concatenate([Q21, Q22], axis=1)]) + \
        np.dot(np.concatenate([Q13, Q23]), F) + \
        np.dot(F.T, np.concatenate([Q31, Q32], axis=1)) + \
        np.dot(np.dot(F.T, Q33), F) + \
        beta * np.dot(np.dot(AF.T, P), AF)

    n = 0
    movieP = np.zeros((2, 2, iternum))
    while (np.abs(nplg.norm(P - P_1)) > 1e-6):
        n = n + 1
        P = P_1
        P11 = P[0:nz, 0:nz]
        P12 = P[0:nz, nz:nz + ns]
        P21 = P12.T
        P22 = P[nz:nz + ns, nz:nz + ns]
        dF = - (Q33 + beta * np.dot(np.dot(B3.T, P22), B3))
        F1 = (Q31 + beta * np.dot(np.dot(B3.T, P21), A) +
              beta * np.dot(np.dot(B3.T, P22), B1)) / dF
        F2 = (Q32 + beta * np.dot(np.dot(B3.T, P22), B2)) / dF
        F = np.concatenate([F1, F2], axis=1)
        AF = np.concatenate([np.concatenate([A.reshape(1, -1), np.zeros((nz, ns))], axis=1),
                             np.concatenate([B1 + B3 * F1, B2 + B3 * F2], axis=1)])
        P_1 = np.concatenate([np.concatenate([Q11, Q12], axis=1),
                              np.concatenate([Q21, Q22], axis=1)]) + \
            np.dot(np.concatenate([Q13, Q23]), F) + \
            np.dot(F.T, np.concatenate([Q31, Q32], axis=1)) + \
            np.dot(np.dot(F.T, Q33), F) + \
            beta * np.dot(np.dot(AF.T, P), AF)
        movieP[:, :, n] = P

    # %check steady state (check whether your code is working properly)
    print(vec[2], np.dot(np.concatenate([F1, F2], axis=1), vec[0:2])[0])

    gkbar = np.dot(np.concatenate([F1, F2], axis=1), np.array(
        [1, kbar]).reshape(-1, 1))[0, 0]
    ckbar = J * np.power(kbar, alpha) + (1.0 - delta) * kbar - gkbar
    ggkbar = np.dot(np.concatenate([F1, F2], axis=1), np.array(
        [1, gkbar]).reshape(-1, 1))[0, 0]
    cgkbar = J * np.power(gkbar, alpha) + (1.0 - delta) * gkbar - ggkbar
    eebar = np.abs(1 - cgkbar / beta / ckbar /
                   (alpha * np.power(gkbar, alpha - 1.0) + 1.0 - delta))

    # %compute Euler equation errors
    kspace = np.linspace(0.5 * kbar, 1.5 * kbar, m_kgrid2).reshape(1, -1)
    gk = np.dot(np.concatenate([F1, F2], axis=1),
                np.concatenate([np.ones(m_kgrid2).reshape(1, -1), kspace]))
    ck = J * np.power(kspace, alpha) + (1.0 - delta) * kspace - gk
    ggk = np.dot(np.concatenate([F1, F2], axis=1),
                 np.concatenate([np.ones(m_kgrid2).reshape(1, -1), gk]))
    cgk = J * np.power(gk, alpha) + (1.0 - delta) * gk - ggk
    ee = np.abs(1 - cgk / beta / ck /
                (alpha * np.power(gk, alpha - 1.0) + 1.0 - delta))

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kspace[0, :], np.log10(np.abs(ee[0, :])))
    title = "LQ1-eulereuqtionerrors.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    fig = plt.figure(figsize=(16, 9))
    kspace2 = np.linspace(0.1 * kbar, 2 * kbar, m_kgrid2)
    m = 0
    num = 200
    Z = np.zeros((m_kgrid2, num))
    for i in range(0, n, n // num + 1):
        temp = movieP[0, 0, i] + 2 * movieP[0, 1, i] * \
            kspace2 + movieP[1, 1, i] * kspace2**2
        plt.plot(kspace2, temp)
        Z[:, m] = temp
        m = m + 1
    title = "LQ1-utilityfunction.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)


if __name__ == "__main__":
    ns = 1
    nz = 1
    nd = 1
    J = 1.0
    alpha = 0.36
    beta = 0.99
    delta = 0.025
    m_kgrid2 = 5000
    iternum = 2000

    figurepath = "../figure"

    LQ1(ns, nz, nd,
        J, alpha, beta, delta,
        m_kgrid2,
        iternum,
        figurepath=figurepath)
