import math
import numpy as np

class CommonCurvedSpace(object):
    def __init__(self, target):
        self.target = target

    def emit_ray(self, pos, alpha):
        g = self.target.metric(pos)

        # dx3 / dx1 = tan(alpha)
        dx1 = -math.cos(alpha) / math.sqrt(-g[1][1])
        dx2 = 0
        dx3 = math.sin(alpha)  / math.sqrt(-g[3][3])

        # dt^2 * g{t,t} - dsp^2  = 0
        ds3 = math.sqrt(-dx1**2 * g[1][1] - dx2**2 * g[2][2] - dx3**2 * g[3][3])
        dx0 = -ds3 / math.sqrt(g[0][0])
        return np.array([dx0, dx1, dx2, dx3])
