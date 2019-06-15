import os
import numpy as np
import numpy.linalg as nplg
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def zje_dp_objective(x,
                     alpha, beta, delta,
                     kgrid,
                     pp, pp2, pp3):
    c = np.power(kgrid, alpha) + (1 - delta) * kgrid - x
    y = - np.sum(np.log(c) + beta * pp(x))
    return y


def zje_dp_objective_jac(x,
                         alpha, beta, delta,
                         kgrid,
                         pp, pp2, pp3):
    c = np.power(kgrid, alpha) + (1 - delta) * kgrid - x
    yp = np.power(c, -1) - beta * pp2(x)
    return yp


def zje_dp_objective_hes(x,
                         alpha, beta, delta,
                         kgrid,
                         pp, pp2, pp3):
    c = np.power(kgrid, alpha) + (1 - delta) * kgrid - x
    ydp = np.diag(np.power(c, -2) - beta * pp3(x))
    return ydp


def zje_dp_pchip(alpha, beta, delta,
                 m_kgrid, m_kgrid2,
                 iternum,
                 figurepath=None):
    '''
    dynamic programming with pchip
    '''
    kbar = np.power(alpha * beta / (1 - beta * (1 - delta)), 1 / (1 - alpha))
    kl = 0.75 * kbar
    kh = 1.25 * kbar
    kgrid = np.linspace(kl, kh, m_kgrid)

    v2 = np.log(kgrid)
    optk2 = kgrid
    lb = np.ones(m_kgrid) * kgrid[0]
    ub = np.power(kgrid, alpha) + (1 - delta) * kgrid - 1e-06

    for kkk in range(iternum):
        v = v2
        optk = optk2
        pp = spi.pchip(kgrid, v)
        pp2 = pp.derivative(1)
        pp3 = pp.derivative(2)

        x0 = optk
        res = spopt.minimize(zje_dp_objective, x0,
                             jac=zje_dp_objective_jac,
                             hess=zje_dp_objective_hes,
                             method="trust-exact",
                             bounds=(lb, ub),
                             args=(alpha, beta, delta, kgrid, pp, pp2, pp3,))
        optk2 = res.x
        optc = np.power(kgrid, alpha) + (1 - delta) * kgrid - optk2
        v2 = np.log(optc) + beta * pp(optk2)
        vdiff = nplg.norm(v2 - v)
        kdiff = nplg.norm(optk2 - optk)
        if (vdiff < 1e-08 and kdiff < 1e-08):
            break

    kspace = np.linspace(kgrid[0], kgrid[m_kgrid - 1], m_kgrid2)
    pp = spi.pchip(kgrid, optk2)
    kp = pp(kspace)
    c = np.power(kspace, alpha) + (1 - delta) * kspace - kp
    kpp = pp(kp)
    cp = np.power(kp, alpha) + (1 - delta) * kp - kpp
    ee = 1 - beta * np.power(cp, -1) * (alpha * np.power(kp,
                                                         alpha - 1) + 1 - delta) / np.power(c, -1)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kspace, np.log10(np.abs(ee)))
    title = "zje_dp_pchip-eulereuqtionerrors.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kspace, kp)
    plt.plot(kspace, kspace, 'k--')
    title = "zje_dp_pchip-k&kp.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    return optk2, kspace, kp, kpp, c, cp, ee


if __name__ == "__main__":
    alpha = 0.36
    beta = 0.99
    delta = 0.025
    m_kgrid = 31
    m_kgrid2 = 10001
    iternum = 2000

    figurepath = "../figure"

    zje_dp_pchip(alpha, beta, delta,
                 m_kgrid, m_kgrid2,
                 iternum,
                 figurepath=figurepath)
