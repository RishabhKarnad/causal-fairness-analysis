import numpy as np


def tv(y, scm):
    prob = 0
    for z in [0, 1]:
        for w in [0, 1]:
            prob += (
                (scm.p_y_zxw(z, 1, w).pmf(y) * scm.p_w_zx(z, 1).pmf(w) * scm.p_x_z(z).pmf(1) * scm.p_z().pmf(
                    z) / np.sum([scm.p_x_z(z_prime).pmf(1) * scm.p_z().pmf(z_prime) for z_prime in [0, 1]]))
                - (scm.p_y_zxw(z, 0, w).pmf(y) * scm.p_w_zx(z, 0).pmf(w) * scm.p_x_z(z).pmf(0) * scm.p_z().pmf(z) / np.sum([scm.p_x_z(z_prime).pmf(0) * scm.p_z().pmf(z_prime) for z_prime in [0, 1]])))
    return prob
