import os
import numpy as np
import numpy.linalg as nplg
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def zje_dp_discrete(alpha, beta, delta,
                    m_kgrid, m_kgrid2,
                    iternum,
                    figurepath=None):
    '''
    discrete dynamic programming
    '''
    kbar = np.power(alpha * beta / (1 - beta * (1 - delta)), 1 / (1 - alpha))
    kl = 0.75 * kbar
    kh = 1.25 * kbar
    kgrid = np.linspace(kl, kh, m_kgrid)

    v2 = np.log(kgrid)
    kgridi = np.repeat(kgrid, m_kgrid).reshape(-1, m_kgrid)
    kgridj = np.repeat(kgrid, m_kgrid).reshape(-1, m_kgrid).T
    u = np.power(kgridi, alpha) + (1 - delta) * kgridi - kgridj
    u[u > 0] = np.log(u[u > 0])
    u[u <= 0] = -1e+10

    optk = kgrid
    imax = np.zeros(m_kgrid)
    for kkk in range(iternum):
        v = v2
        vhat = np.repeat(v, m_kgrid).reshape(-1, m_kgrid).T
        temp = u.T + beta * vhat.T
        v2 = temp.max(axis=0)
        imax = temp.argmax(axis=0)
        optk2 = kgrid[imax]
        vdiff = nplg.norm(v2 - v)
        if (vdiff < 1e-08 and kdiff < 1e-08):
            break

    kspace = np.linspace(kgrid[0], kgrid[m_kgrid - 1], m_kgrid2)
    pp1d = spi.interp1d(kgrid, optk2, kind="nearest")
    kp = pp1d(kspace)
    c = np.power(kspace, alpha) + (1 - delta) * kspace - kp
    kpp = pp1d(kp)
    cp = np.power(kp, alpha) + (1 - delta) * kp - kpp
    ee = 1 - beta * np.power(cp, -1) * (alpha * np.power(kp,
                                                         alpha - 1) + 1 - delta) / np.power(c, -1)

    fig = plt.figure(figsize=(16, 9))
    temp = ee.copy()
    temp[temp <= 0] = 1e-10
    plt.plot(kspace, np.log10(np.abs(temp)))
    title = "zje_dp_discrete-eulereuqtionerrors.png"
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    if figurepath is not None:
        fig.savefig(os.path.join(figurepath, title), dpi=300)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(kspace, kp)
    plt.plot(kspace, kspace, 'k--')
    title = "zje_dp_discrete-k&kp.png"
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
    m_kgrid = 1001
    m_kgrid2 = 10001
    iternum = 2000

    figurepath = "../figure"

    zje_dp_discrete(alpha, beta, delta,
                    m_kgrid, m_kgrid2,
                    iternum,
                    figurepath=figurepath)
