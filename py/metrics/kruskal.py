import math
import scipy.special

import metrics.common

import numpy as np

def W0(z):
    return scipy.special.lambertw(z).real

def dWdz(z):
    if z < -1 + 1e-5:
        z = -1 + 1e-6
    return 1 / (z + math.exp(W0(z)))

def radius_relative(T, X):
    return 1 + W0((X**2 - T**2) / math.e)

class KruskalSpace(object):
    def __init__(self, rs):
        self.rs = rs
        self.common = metrics.common.CommonCurvedSpace(self)

    def metric(self, pos):
        T = pos[0]
        X = pos[1]
        theta = pos[2]
        phi = pos[3]

        rr = radius_relative(T, X)
        k = 4 * self.rs**2 / rr * math.exp(-rr)

        return np.diag([k, -k, -(rr*self.rs)**2, -(rr*self.rs*math.sin(theta))**2])

    def transform_to(self, pos, dir):
        t = pos[0]
        r = pos[1]

        k = r/self.rs - 1
        exp = math.exp(r/(2*self.rs))
        sinh = math.sinh(t/(2*self.rs))
        cosh = math.cosh(t/(2*self.rs))

        if dir is not None:
            dt = dir[0]
            dr = dir[1]

            dk = 1/rs * dr
            dexp = 1/(2*rs) * exp * dr
            dsinh = 1/(2*rs) * cosh * dt
            dcosh = 1/(2*rs) * sinh * dt

        if r > self.rs:
            T = k**0.5 * exp * sinh
            X = k**0.5 * exp * cosh

            if dir is not None:
                dT = 0.5 * k**(-0.5) * dk * exp * sinh + k**0.5 * dexp * sinh + k**0.5 * exp * dsinh
                dX = 0.5 * k**(-0.5) * dk * exp * cosh + k**0.5 * dexp * cosh + k**0.5 * exp * dcosh
                return True, True, (T, X, pos[2], pos[3]), (dT, dX, dir[2], dir[3])
            else:
                return True, False, (T, X, pos[2], pos[3]), None
        else:
            X = (-k)**0.5 * exp * sinh
            T = (-k)**0.5 * exp * cosh
            return True, False, (T, X, pos[2], pos[3]), None

    def transform_from(self, pos, dir):
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

        r = radius_relative(T, X) * self.rs

        if (abs(T) < abs(X)):
            # outside BH
            t = 2 * self.rs * math.atanh(T/X)
        else:
            # inside BH
            t = 2 * self.rs * math.atanh(X/T)

        if r <= self.rs:
            return True, False, (t, r, pos[2], pos[3]), None, attrs
        else:
            if dir is not None:
                dt = 2*self.rs * (X*dT - T*dX) / (X**2-T**2)
                dr = 2*self.rs / math.e * dWdz((X**2-T**2)/math.e) * (X*dX - T*dT)
                return True, True, (t, r, pos[2], pos[3]), (dt, dr, dir[2], dir[3]), attrs
            else:
                return True, False, (t, r, pos[2], pos[3]), None, attrs

    def emit_ray(self, t0, r0, alpha):
        theta = math.pi / 2
        phi = 0

        pos_valid, _, pos, _ = self.transform_to([t0, r0, theta, phi], None)
        if not pos_valid:
            return False, None, None
        dir = self.common.emit_ray(pos, alpha)
        return True, pos, dir

    def check_collision(self, pos):
        T = pos[0]
        X = pos[1]
        rr = radius_relative(T, X)
        if rr < 2e-3:
            return True
        return False
