import math

def transform_to(pos, dir, rs):
    return True, pos, dir

def transform_from(pos, dir, rs):
    return True, pos, dir, {}

def ray(r0, alpha, rs):
    k = 1 - rs/r0
    sqrtk = k**0.5
    dr   = -math.cos(alpha)
    dphi = math.sin(alpha) / (r0*sqrtk)

    dl = (1/k * dr**2 + r0**2 * dphi**2)**0.5
    dt = -dl / sqrtk

    return True, (0, r0, math.pi/2, 0), (dt, dr, 0, dphi)

def collision(pos, rs):
    r = pos[1]
    if r < 1.05 * rs:
        return True
    return False
