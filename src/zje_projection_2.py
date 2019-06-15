import os
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util import GaussHermite


def geta_zju_2(x,
               alpha, beta, delta, rho, sigmaa,
               kgrid, kweights, agrid, aweights,
               enodes, eweights,
               m_gh=None,
               is_opt=True):
    '''
    using projection to solve growth model
    '''
    mk = kgrid.shape[0]
    ma = agrid.shape[0]
    if m_gh is None:
        m_gh = ma
    kgrid = np.repeat(kgrid, ma).reshape(mk, ma)
    agrid = np.repeat(agrid, mk).reshape(ma, mk).T
    kp = x[0] + x[1] * kgrid + x[2] * agrid + \
        x[3] * (2 * kgrid ** 2 - 1) + \
        x[4] * kgrid * agrid + \
        x[5] * (2 * agrid ** 2 - 1)
    c = np.exp(agrid) * np.power(kgrid, alpha) + \
        (1 - delta) * kgrid - kp

    esum = np.zeros((mk, ma))
    for k in range(m_gh):
        ap = rho * agrid + enodes[k]
        kpp = x[0] + x[1] * kp + x[2] * ap + \
            x[3] * (2 * kp * kp - 1) + x[4] * kp * ap + \
            x[5] * (2 * ap * ap - 1)
        cp = np.exp(ap) * np.power(kp, alpha) + (1 - delta) * kp - kpp
        esum = esum + eweights[k] * np.power(cp, -1) * \
            (alpha * np.exp(ap) * np.power(kp, alpha - 1) + 1 - delta)
    r = np.power(c, -1) - beta * esum

    if is_opt:
        kweights = np.repeat(kweights, ma).reshape(mk, ma)
        aweights = np.repeat(aweights, mk).reshape(ma, mk).T
        y = (kweights * aweights * (r ** 2)).sum()
        return y
    else:
        return kp, kpp, c, cp, r


def zje_projection_2(alpha, beta, delta, rho, sigmaa,
                     mk_grid, ma_grid,
                     mk_grid2, ma_grid2, m_gh,
                     figurepath=None):
    '''
    using projection to solve growth model
    '''
    kbar = np.power(alpha * beta / (1 - beta * (1 - delta)), 1 / (1 - alpha))
    kl = 0.75 * kbar
    kh = 1.25 * kbar
    i = np.arange(mk_grid) + 1
    zk = - np.cos((2 * i - 1) / (2 * mk_grid) * np.pi)
    kgrid = (zk + 1) * (kh - kl) / 2 + kl
    kweights = np.pi * np.ones(mk_grid) / mk_grid

    al = -3 * sigmaa / np.sqrt(1 - rho * rho)
    ah = 3 * sigmaa / np.sqrt(1 - rho * rho)
    za = np.zeros(ma_grid)
    agrid = np.zeros(ma_grid)
    i = np.arange(ma_grid) + 1
    za = - np.cos((2 * i - 1) / (2 * ma_grid) * np.pi)
    agrid = (za + 1) * (ah - al) / 2 + al
    aweights = np.pi * np.ones(ma_grid) / ma_grid

    enodes, eweights = GaussHermite(ma_grid)
    eweights = eweights / np.sum(eweights)
    enodes = np.sqrt(2) * sigmaa * enodes

    x0 = np.array([kbar * (1 - 0.96), 0.96, 0.1, 0.0, 0.0, 0.0])
    # Nelder-Mead method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html#scipy.optimize.fmin
    x = spopt.fmin(geta_zju_2, x0,
                   args=(alpha, beta, delta, rho, sigmaa,
                         kgrid, kweights, agrid, aweights,
                         enodes, eweights))

    kgrid2 = np.linspace(kgrid[0], kgrid[mk_grid - 1], mk_grid2)
    agrid2 = np.linspace(agrid[0], agrid[ma_grid - 1], ma_grid2)
    enodes2, eweights2 = GaussHermite(m_gh)
    eweights2 = eweights2 / np.sum(eweights2)
    enodes2 = np.sqrt(2) * sigmaa * enodes2

    kp, kpp, c, cp, ee = geta_zju_2(x,
                                    alpha, beta, delta, rho, sigmaa,
                                    kgrid2, kweights, agrid2, aweights,
                                    enodes2, eweights2,
                                    m_gh=m_gh,
                                    is_opt=False)

    fig = plt.figure(figsize=(16, 9))
    plt.contourf(agrid2, kgrid2, np.log10(np.abs(ee)), m_gh)
    title = "zje_projection_2-eulereuqtionerrors.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    X, Y = np.meshgrid(agrid2, kgrid2)
    kp_45 = np.repeat(kgrid2, mk_grid2).reshape(mk_grid2, -1)
    fig = plt.figure(figsize=(16, 16))
    title = "zje_projection_2-k&kp.png"
    plt.title(title, fontsize=20)
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_surface(X, Y, kp)
    ax.plot_surface(X, Y, kp_45)
    ax.view_init(10, 10)
    ax = fig.add_subplot(212, projection='3d')
    ax.plot_surface(X, Y, kp)
    ax.plot_surface(X, Y, kp_45)
    ax.view_init(30, 10)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    return x, agrid2, kgrid2, enodes2, eweights2, kp, kpp, c, cp, ee


if __name__ == "__main__":
    alpha = 0.36
    beta = 0.99
    delta = 0.025
    rho = 0.95
    sigmaa = 0.0076
    mk_grid = 11
    ma_grid = 11
    mk_grid2 = 1001
    ma_grid2 = 1001
    m_gh = 31

    figurepath = "../figure"

    x, agrid2, kgrid2, enodes2, eweights2, kp, kpp, c, cp, ee = zje_projection_2(alpha, beta, delta, rho, sigmaa,
                                                                                 mk_grid, ma_grid,
                                                                                 mk_grid2, ma_grid2, m_gh,
                                                                                 figurepath=figurepath)
