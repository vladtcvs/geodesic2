import math

def radius(tau, rho, rs):
    return ((3/2*(rho-tau)) ** (2/3)) * (rs ** (1/3))

def transform_to(pos, dir, rs):
    t = pos[0]
    r = pos[1]

    tau = t
    rho = tau + 2/3*(r**1.5)/(rs**0.5)

    if r < rs:
        return False, (tau, rho, pos[2], pos[3]), None

    a = (rs/r)**0.5 / (1 - rs/r)
    b = (r/rs)**0.5 / (1 - rs/r)

    dt = dir[0]
    dr = dir[1]
    dtau = dt + a * dr
    drho = dt + b * dr

    return True, (tau, rho, pos[2], pos[3]), (dtau, drho, dir[2], dir[3])

def transform_from(pos, dir, rs):
    tau = pos[0]
    rho = pos[1]

    r = radius(tau, rho, rs)
    t = 0

    if r < rs:
        return False, (t, r, pos[2], pos[3]), None

    dtau = dir[0]
    drho = dir[1]

    a = (rs/r)**0.5 / (1 - rs/r)
    b = (r/rs)**0.5 / (1 - rs/r)
    D = b - a

    dt = 1/D * (b*dtau - a * drho)
    dr = 1/D * (-dtau + drho)

    return True, (t, r, pos[2], pos[3]), (dt, dr, dir[2], dir[3]), {}

def ray(r0, alpha, rs):

    tau = 0
    rho = tau + 2/3*(r0**1.5)/(rs**0.5)

    # dtau ** 2 - rs/r * drho**2 - r**2 * dphi**2 = 0

    drho = -math.cos(alpha)
    dphi = math.sin(alpha) * (rs / r0**3)**0.5
    dtau = -(rs/r0*drho**2 + r0**2 * dphi**2)**0.5

    return True, (tau, rho, math.pi/2, 0), (dtau, drho, 0, dphi)

def collision(pos, rs):
    tau = pos[0]
    rho = pos[1]

    r = ((3/2*(rho-tau)) ** (2/3)) * (rs ** (1/3))
    if abs(r - rs)/rs < 1.5e-6:
        return True
    return False
