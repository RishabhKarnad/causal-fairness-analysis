import numpy as np

from tqdm import tqdm


class StandardModelSCM:
    def __init__(self, d_z=1, d_w=1):
        self.d_z = d_z
        self.d_w = d_w

    def p_z(self):
        raise NotImplementedError

    def p_x_z(self, z):
        raise NotImplementedError

    def p_w_zx(self, z, x):
        raise NotImplementedError

    def p_y_zxw(self, z, x, w):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def sample_n(self, n_samples=1000):
        samples = []
        for i in tqdm(range(n_samples)):
            samples.append(self.sample())
        return np.array(samples)
