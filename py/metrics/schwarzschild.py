import math
import numpy as np

import metrics.common as common

class SchwarzschildSpace(object):
    def __init__(self, rs):
        self.rs = rs
        self.common = common.CommonCurvedSpace(self)

    def metric(self, pos):
        r = pos[1]
        theta = pos[2]
        
        k = 1 - self.rs/r
        sqrtk = k**0.5
        return np.diag([k, -1/k, -r**2, -(r**2) * math.sin(theta)**2])

    def transform_to(self, pos, dir):
        return True, True, pos, dir

    def transform_from(self, pos, dir):
        return True, True, pos, dir, {}

    def emit_ray(self, t0, r0, alpha):
        pos = np.array([t0, r0, math.pi/2, 0])
        dir = self.common.emit_ray(pos, alpha)
        return True, pos, dir

    def check_collision(self, pos):
        r = pos[1]
        if r < 1.05 * self.rs:
            return True
        return False
