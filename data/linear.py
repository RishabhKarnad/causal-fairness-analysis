import numpy as np
import scipy.stats as stats

from .scm import StandardModelSCM


class FairGaussianSCM(StandardModelSCM):
    def p_z(self):
        mu_z = 0.75
        return stats.norm(mu_z, 0.1)

    def p_x_z(self, z):
        mu_x_z = 2*z + 10
        return stats.norm(mu_x_z, 0.1)

    def p_w_zx(self, z, x):
        mu_w_zx = 5*x - 12
        return stats.norm(mu_w_zx, 0.1)

    def p_y_zxw(self, z, x, w):
        return stats.bernoulli(1/(1+(np.exp(0.01*(-2*z+1)))))

    def sample(self):
        z = self.p_z().rvs()
        x = self.p_x_z(z).rvs()
        w = self.p_w_zx(None, x).rvs()
        y = self.p_y_zxw(z, x, w).rvs()
        return np.array([z, x, w, y])


class UnfairGaussianSCM(StandardModelSCM):
    def p_z(self):
        mu_z = 15
        return stats.norm(mu_z, 0.1)

    def p_x_z(self, z):
        mu_x_z = 2*z + 10
        return stats.norm(mu_x_z, 0.1)

    def p_w_zx(self, z, x):
        mu_w_zx = 5*x - 12
        return stats.norm(mu_w_zx, 0.1)

    def p_y_zxw(self, z, x, w):
        return stats.bernoulli(1/(1+np.exp(0.001*(-10*z+3*x+2*w+1))))

    def sample(self):
        z = self.p_z().rvs()
        x = self.p_x_z(z).rvs()
        w = self.p_w_zx(None, x).rvs()
        y = self.p_y_zxw(z, x, w).rvs()
        return np.array([z, x, w, y])
