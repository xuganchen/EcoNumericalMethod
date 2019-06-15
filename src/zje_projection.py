import os
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt


def geta_zju(x,
             alpha, beta, delta,
             kgrid, weights,
             is_opt=True):
    m = kgrid.shape[0]
    kp = x[0] + x[1] * kgrid + x[2] * (2 * kgrid ** 2 - 1) + \
        x[3] * (4 * kgrid ** 3 - 3 * kgrid)
    c = np.power(kgrid, alpha) + (1 - delta) * kgrid - kp
    kpp = x[0] + x[1] * kp + x[2] * (2 * kp ** 2 - 1) + \
        x[3] * (4 * kp ** 3 - 3 * kp)
    cp = np.power(kp, alpha) + (1 - delta) * kp - kpp
    r = np.power(c, -1) - beta * np.power(cp, -1) * \
        (alpha * np.power(kp, alpha - 1) + 1 - delta)
    if is_opt:
        y = np.sum(r * weights * r)
        return y
    else:
        return kp, kpp, c, cp, r


def zje_projection(alpha, beta, delta,
                   m_kgrid, m_kgrid2,
                   figurepath=None):
    '''
    using projection to solve growth model
    '''
    kbar = np.power(alpha * beta / (1 - beta * (1 - delta)), 1 / (1 - alpha))
    kl = 0.75 * kbar
    kh = 1.25 * kbar
    i = np.arange(m_kgrid) + 1
    z = - np.cos((2 * i - 1) / (2 * m_kgrid) * np.pi)
    kgrid = (z + 1) * (kh - kl) / 2 + kl
    weights = np.pi * np.ones(m_kgrid) / m_kgrid

    x0 = np.array([kbar * (1 - 0.96), 0.96, 0.0, 0.0])
    # Nelder-Mead method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html#scipy.optimize.fmin
    x = spopt.fmin(geta_zju, x0,
                   args=(alpha, beta, delta, kgrid, weights, ))

    kgrid2 = np.linspace(kgrid[0], kgrid[m_kgrid - 1], m_kgrid2)
    kp, kpp, c, cp, ee = geta_zju(x,
                                  alpha, beta, delta,
                                  kgrid2, weights,
                                  is_opt=False)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kgrid2, np.log10(np.abs(ee)))
    title = "zje_projection-eulereuqtionerrors.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kgrid2, kp)
    plt.plot(kgrid2, kgrid2, 'k--')
    title = "zje_projection-k&kp.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    return x, kgrid2, kp, kpp, c, cp, ee


if __name__ == "__main__":
    alpha = 0.36
    beta = 0.99
    delta = 0.025
    m_kgrid = 31
    m_kgrid2 = 10001

    figurepath = "../figure"

    kgrid2, kp, kpp, c, cp, ee = zje_projection(
        alpha, beta, delta, m_kgrid, m_kgrid2,
        figurepath=figurepath)
