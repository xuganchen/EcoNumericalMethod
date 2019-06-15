import sympy as sy


def anal_deriv(approx, config):
    '''
    # This program copmutes analytical first and second (if approx=2) derivatives 
    # of the function f(yp,y,xp,x) with respect to x, y, xp, and yp.  
    # For documentation, see the paper ``Solving Dynamic General Equilibrium 
    # Models Using a Second-Order Approximation to the Policy Function,'' 
    # by Stephanie Schmitt-Grohe and Martin Uribe (JEDC, vol. 28, January 2004, pp. 755-775). 

    # Inputs: f, x, y, xp, yp, approx
    # Output: Analytical first and second derivatives of f. 

    # If approx is set at a value different from 2, the program delivers 
    # the first derivatives of f and sets second derivatives at zero. 
    # If approx equals 2, the program returns first and second 
    # derivatives of f. The default value of approx is 2. 
    # Note: This program requires MATLAB's Symbolic Math Toolbox
    '''
    n = config['n']

    for i in ['x', 'xp', 'y', 'yp']:
        if approx > 0:
            ni = config['n' + i]
            f = config['f']
            temp = sy.Array(f.jacobian(config[i])).reshape(n, ni)
        else:
            temp = 0
        config['f' + i] = temp

    for i in ['x', 'xp', 'y', 'yp']:
        for j in ['x', 'xp', 'y', 'yp']:
            if approx > 1:
                ni = config['n' + i]
                nj = config['n' + j]
                fi = config['f' + i]
                vj = config[j]
                temp = sy.Array(sy.Matrix(fi[:]).jacobian(
                    vj)).reshape(n, ni, nj)
            else:
                temp = 0
            config['f' + i + j] = temp

    for i in ['x', 'xp', 'y', 'yp']:
        for j in ['x', 'xp', 'y', 'yp']:
            for k in ['x', 'xp', 'y', 'yp']:
                if approx > 2:
                    ni = config['n' + i]
                    nj = config['n' + j]
                    nk = config['n' + k]
                    fij = config['f' + i + j]
                    vk = config[k]
                    temp = sy.Array(sy.Matrix(fij[:]).jacobian(
                        vk)).reshape(n, ni, nj, nk)
                else:
                    temp = 0
                config['f' + i + j + k] = temp

    return config


def growth_model_zje(approx,
                     is_log=False):
    config = {}
    config['approx'] = approx
    config['is_log'] = is_log

    # Define parameters
    delta, betta, alfa = sy.symbols("delta, betta, alfa")
    for i in ["delta", "betta", "alfa"]:
        config[i] = vars()[i]

    # Define variables
    c, k, invt = sy.symbols("c, k, invt")
    cp, kp, invtp = sy.symbols("cp, kp, invtp")
    for i in ["c", "k", "invt", "cp", "kp", "invtp"]:
        config[i] = vars()[i]

    # Write equations
    # resource contraint
    f1 = k**alfa - c - invt
    # investment
    f2 = invt + (1 - delta) * k - kp
    # intertemporal FOC
    f3 = c**(-1) - betta * cp**(-1) * (1 + alfa * kp**(alfa - 1) - delta)

    # Create function f
    f = sy.Matrix([[f1], [f2], [f3]])

    # Define the vector of controls, y, and states, x
    constant = sy.Array([delta, betta, alfa])
    x = sy.Array([k])
    y = sy.Array([c, invt])
    xp = sy.Array([kp])
    yp = sy.Array([cp, invtp])

    # Make f a function of the logarithm of the state and control vector
    # --  remember to change steady state and data accordingly
    if is_log:
        subs_dict = {}
        for i in [c, k, invt, cp, kp, invtp]:
            subs_dict[i] = sy.exp(i)
        f = f.subs(subs_dict)

    for i in ['f', 'x', 'xp', 'y', 'yp', 'constant']:
        config[i] = vars()[i]
    config['nx'] = x.shape[0]
    config['ny'] = y.shape[0]
    config['nxp'] = xp.shape[0]
    config['nyp'] = yp.shape[0]
    config['n'] = f.shape[0]

    config = anal_deriv(approx=approx, config=config)

    return config


def growth_model_zje_ss(config):
    '''
    GROWTH_MODEL_SS.M
    This program computes the steady state of the model 
    '''
    nbetta = config['nbetta']
    nalfa = config['nalfa']
    ndelta = config['ndelta']
    is_log = config['is_log']

    # Steady state in levels
    nk = (ndelta * nbetta / (1 - nbetta * (1 - ndelta)))**(1 / (1 - ndelta))
    ninvt = ndelta * nk
    nc = nk**ndelta - ndelta * nk

    if is_log:
        nk = sy.log(nk)
        ninvt = sy.log(ninvt)
        nc = sy.log(nc)

    nkp = nk
    ninvtp = ninvt
    ncp = nc

    for i in ["nk", "ninvt", "nc", "nkp", "ninvtp", "ncp"]:
        config[i] = vars()[i]

    return config


if __name__ == "__main__":
    approx = 3
    is_log = False
    config = growth_model_zje(approx, is_log=is_log)
    config = growth_model_zje_ss(config)
    print(config)
