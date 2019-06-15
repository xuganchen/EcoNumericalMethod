import os
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def geta_zju_pchip(x,
                   alpha, beta, delta,
                   kgrid,
                   ppgrid=None,
                   is_opt=True):
    m = kgrid.shape[0]
    if ppgrid is None:
        ppgrid = kgrid
    pp = spi.pchip(ppgrid, x)
    kp = pp(kgrid)
    c = np.power(kgrid, alpha) + (1 - delta) * kgrid - kp
    kpp = pp(kp)
    cp = np.power(kp, alpha) + (1 - delta) * kp - kpp
    r = np.power(c, -1) - beta * np.power(cp, -1) * \
        (alpha * np.power(kp, alpha - 1) + 1 - delta)
    if is_opt:
        return r
    else:
        return pp, kp, kpp, c, cp, r


def zje_projection_pchip(alpha, beta, delta,
                         m_kgrid, m_kgrid2,
                         figurepath=None):
    '''
    using projection to solve growth model
    '''
    kbar = np.power(alpha * beta / (1 - beta * (1 - delta)), 1 / (1 - alpha))
    kl = 0.75 * kbar
    kh = 1.25 * kbar
    kgrid = np.linspace(kl, kh, m_kgrid)

    x0 = kgrid
    x = spopt.fsolve(geta_zju_pchip, x0,
                     args=(alpha, beta, delta, kgrid, ))

    kgrid2 = np.linspace(kgrid[0], kgrid[m_kgrid - 1], m_kgrid2)
    pp, kp, kpp, c, cp, ee = geta_zju_pchip(x,
                                            alpha, beta, delta,
                                            kgrid2,
                                            ppgrid=kgrid,
                                            is_opt=False)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kgrid2, np.log10(np.abs(ee)))
    title = "zje_projection_pchip-eulereuqtionerrors.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kgrid2, kp)
    plt.plot(kgrid2, kgrid2, 'k--')
    title = "zje_projection_pchip-k&kp.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    return x, kgrid, kgrid2, pp, kp, kpp, c, cp, ee


if __name__ == "__main__":
    alpha = 0.36
    beta = 0.99
    delta = 0.025
    m_kgrid = 31
    m_kgrid2 = 10001

    figurepath = "../figure"

    x, kgrid, kgrid2, pp, kp, kpp, c, cp, ee = zje_projection_pchip(alpha, beta, delta,
                                                                    m_kgrid, m_kgrid2,
                                                                    figurepath=figurepath)
