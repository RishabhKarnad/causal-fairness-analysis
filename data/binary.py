import numpy as np
import scipy.stats as stats

from .scm import StandardModelSCM


class FairSCM(StandardModelSCM):
    def p_z(self):
        return stats.bernoulli(3/4)

    def p_x_z(self, z):
        return stats.bernoulli(2/3) if z == 0 else stats.bernoulli(1/4)

    def p_w_zx(self, z, x):
        return stats.bernoulli(1/2) if x == 0 else stats.bernoulli(3/4)

    def p_y_zxw(self, z, x, w):
        return stats.bernoulli(1/(1+np.exp(-2*z+1)))

    def sample(self):
        z = self.p_z().rvs()
        x = self.p_x_z(z).rvs()
        w = self.p_w_zx(None, x).rvs()
        y = self.p_y_zxw(z, x, w).rvs()
        return [z, x, w, y]


class UnfairSCM(StandardModelSCM):
    def p_z(self):
        return stats.bernoulli(3/4)

    def p_x_z(self, z):
        return stats.bernoulli(2/3) if z == 0 else stats.bernoulli(1/4)

    def p_w_zx(self, z, x):
        return stats.bernoulli(1/2) if x == 0 else stats.bernoulli(3/4)

    def p_y_zxw(self, z, x, w):
        return stats.bernoulli(1/(1+np.exp(-10*z+3*x+2*w+1)))

    def sample(self):
        z = self.p_z().rvs()
        x = self.p_x_z(z).rvs()
        w = self.p_w_zx(None, x).rvs()
        y = self.p_y_zxw(z, x, w).rvs()
        return [z, x, w, y]
