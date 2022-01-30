import math
import scipy.special

import schwarzschild

def W0(x):
    return scipy.special.lambertw(x).real

def dWdz(z):
    if z < -1 + 1e-5:
        z = -1 + 1e-6
    return 1 / (z + math.exp(W0(z)))

def radius_relative(T, X):
    return 1 + W0((X**2 - T**2) / math.e)

def transform_to(pos, dir, rs):
    t = pos[0]
    r = pos[1]

    k = r/rs - 1
    exp = math.exp(r/(2*rs))
    sinh = math.sinh(t/(2*rs))
    cosh = math.cosh(t/(2*rs))


    if dir is not None:
        dt = dir[0]
        dr = dir[1]

        dk = 1/rs * dr
        dexp = 1/(2*rs) * exp * dr
        dsinh = 1/(2*rs) * cosh * dt
        dcosh = 1/(2*rs) * sinh * dt

    if r > rs:
        T = k**0.5 * exp * sinh
        X = k**0.5 * exp * cosh

        if dir is not None:
            dT = 0.5 * k**(-0.5) * dk * exp * sinh + k**0.5 * dexp * sinh + k**0.5 * exp * dsinh
            dX = 0.5 * k**(-0.5) * dk * exp * cosh + k**0.5 * dexp * cosh + k**0.5 * exp * dcosh

    else:
        X = (-k)**0.5 * exp * sinh
        T = (-k)**0.5 * exp * cosh

        if dir is not None:
            # TODO:
            dT = 0
            dX = 0

    if dir is not None:
        return True, (T, X, pos[2], pos[3]), (dT, dX, dir[2], dir[3])
    else:
        return True, (T, X, pos[2], pos[3]), None

def transform_from(pos, dir, rs):
    T = pos[0]
    X = pos[1]

    if dir is not None:
        dT = dir[0]
        dX = dir[1]

    attrs = {}
    if (X**2 - T**2) > 0:
        if X > 0:
            attrs["area"] = 1
        else:
            attrs["area"] = 3
    else:
        if T > 0:
            attrs["area"] = 2
        else:
            attrs["area"] = 4

    r = radius_relative(T, X) * rs

    if (abs(T) < abs(X)):
        # outside BH
        t = 2 * rs * math.atanh(T/X)
    else:
        # inside BH
        t = 2 * rs * math.atanh(X/T)

    if (r <= rs):
        return False, (t, r, pos[2], pos[3]), None, attrs

    if dir is not None:
        dt = 2*rs * (X*dT - T*dX) / (X**2-T**2)
        dr = 2*rs / math.e * dWdz((X**2-T**2)/math.e) * (X*dX - T*dT)
        return True, (t, r, pos[2], pos[3]), (dt, dr, dir[2], dir[3]), attrs
    else:
        return True, (t, r, pos[2], pos[3]), None, attrs

def ray_schwarzschild(r0, alpha, rs):
    res, pos, dir = schwarzschild.ray(r0, alpha, rs)
    return transform_to(pos, dir, rs)

def ray_xt(r0, alpha, rs):
    t = 0
    theta = math.pi / 2
    phi = 0

    k = 4 * rs*rs*rs / r0 * math.exp(-r0/rs)

    _, pos, _ = transform_to((t, r0, theta, phi), None, rs)
    T, X, _, _ = pos

    dX = -math.cos(alpha)
    dtheta = 0
    dphi = math.sin(alpha) * math.sqrt(k) / r0

    # dT**2 = dX**2 + r0**2 / k * dphi**2
    dT = -(dX*dX + r0**2 / k * dphi**2)**0.5

    return True, (T, X, theta, phi), (dT, dX, dtheta, dphi)

def ray(r0, alpha, rs):
    return ray_xt(r0, alpha, rs)
#    return ray_schwarzschild(r0, alpha, rs)

def collision(pos, rs):
    T = pos[0]
    X = pos[1]
    r = radius_relative(T, X)
    if r < 2e-3:
        return True
    return False
