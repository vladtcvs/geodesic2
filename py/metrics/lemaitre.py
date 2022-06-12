import math
import numpy as np

import metrics.common as common

class LemaitreSpace(object):
    def __init__(self, rs):
        self.rs = rs
        self.common = common.CommonCurvedSpace(self)

    def radius(self, tau, rho, rs):
        return ((3/2*(rho-tau)) ** (2/3)) * (rs ** (1/3))

    def metric(self, pos):
        tau   = pos[0]
        rho   = pos[1]
        theta = pos[2]
        phi   = pos[3]
        r = self.radius(tau, rho, self.rs)
        return np.diag([1, -self.rs/r, -r**2, -r**2 * math.sin(theta)**2])
    
    def transform_to(self, pos, dir):
        t = pos[0]
        r = pos[1]

        tau = t
        rho = tau + 2/3*(r**1.5)/(self.rs**0.5)

        if r < self.rs:
            return True, False, (tau, rho, pos[2], pos[3]), None

        if dir is None:
            return True, False, (tau, rho, pos[2], pos[3]), None

        a = (self.rs/r)**0.5 / (1 - self.rs/r)
        b = (r/self.rs)**0.5 / (1 - self.rs/r)

        dt = dir[0]
        dr = dir[1]
        dtau = dt + a * dr
        drho = dt + b * dr

        return True, True, (tau, rho, pos[2], pos[3]), (dtau, drho, dir[2], dir[3])

    def transform_from(self, pos, dir):
        tau = pos[0]
        rho = pos[1]

        r = self.radius(tau, rho, self.rs)
        t = 0 # TODO: fix it

        if r < self.rs:
            return False, False, (t, r, pos[2], pos[3]), None, {}

        if dir is None:
            return True, False, (t, r, pos[2], pos[3]), None, {}

        dtau = dir[0]
        drho = dir[1]

        a = (self.rs/r)**0.5 / (1 - self.rs/r)
        b = (r/self.rs)**0.5 / (1 - self.rs/r)
        D = b - a

        dt = 1/D * (b*dtau - a * drho)
        dr = 1/D * (-dtau + drho)

        return True, True, (t, r, pos[2], pos[3]), (dt, dr, dir[2], dir[3]), {}

    def emit_ray(self, t0, r0, alpha):
        pos_valid, _, pos, _ = self.transform_to([t0, r0, math.pi/2, 0], None)
        if not pos_valid:
            return False, pos, None
        dir = self.common.emit_ray(pos, alpha)
        # dtau ** 2 - rs/r * drho**2 - r**2 * dphi**2 = 0
        return True, pos, dir

    def check_collision(self, pos):
        pos_valid, _, pos_s, _, _ = self.transform_from(pos, None)
        if not pos_valid:
            return True
        r = pos_s[1]        
        if abs(r - self.rs)/self.rs < 1.5e-6:
            return True
        return False
