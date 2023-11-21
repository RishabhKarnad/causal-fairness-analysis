import numpy as np
import scipy.stats as stats

from .scm import StandardModelSCM


class FairNonlinearGaussianSCM(StandardModelSCM):
    def p_z(self):
        mu_z = 15
        return stats.norm(mu_z, 0.1)

    def p_x_z(self, z):
        mu_x_z = 2*z**3 - 14*z + 10
        return stats.norm(mu_x_z, 0.1)

    def p_w_zx(self, z, x):
        mu_w_zx = 5*x**2 - 12
        return stats.norm(mu_w_zx, 0.1)

    def p_y_zxw(self, z, x, w):
        a = 1/(1 + np.exp((-1*z + 15.25)))
        #print(a)
        return 0 if a < 0.425 else 1 #stats.bernoulli(1/(1+np.exp(0.01*(-2*z+1))))

    def sample(self):
        z = self.p_z().rvs()
        x = self.p_x_z(z).rvs()
        w = self.p_w_zx(None, x).rvs()
        y = self.p_y_zxw(z, x, w)#.rvs()
        return np.array([z, x, w, y])


class UnfairNonlinearGaussianSCM(StandardModelSCM):
    def __init__(self, d_z=1, d_w=1, alpha=0.01):
        super().__init__(d_z, d_w)
        self.alpha = alpha

    def p_z(self):
        mu_z = 15
        return stats.norm(mu_z, 0.2)

    def p_x_z(self, z):
        mu_x_z = 2*z**3 - 14*z + 10
        return stats.norm(mu_x_z, 0.2)

    def p_w_zx(self, z, x):
        mu_w_zx = 5*x**2 - 12
        return stats.norm(mu_w_zx, 0.2)

    def p_y_zxw(self, z, x, w):
        a = 1/(1+np.exp(0.1*(-2*z+0.00000013*x-0.000000002*w+25)))
        #print(a)
        return 0 if a < 0.625 else 1 #stats.bernoulli(1/(1+np.exp(0.001*(-10*z+0.00000013*x-0.000000002*w+1))))

    def sample(self):
        z = self.p_z().rvs()
        x = self.p_x_z(z).rvs()
        w = self.p_w_zx(None, x).rvs()
        y = self.p_y_zxw(z, x, w)#.rvs()
        return np.array([z, x, w, y])
