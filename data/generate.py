import numpy as np

from data.linear import FairGaussianSCM, UnfairGaussianSCM
from data.nonlinear import FairNonlinearGaussianSCM, UnfairNonlinearGaussianSCM


def gen_data(scm, n_samples=1000):
    return scm.sample_n(n_samples)


def generate():
    filenames = ['fair_linear', 'unfair_linear',
                 'fair_nonlinear', 'unfair_nonlinear']
    models = [FairGaussianSCM, UnfairGaussianSCM,
              FairNonlinearGaussianSCM, UnfairNonlinearGaussianSCM]
    # models = [FairGaussianSCM, UnfairGaussianSCM]
    for idx, model in enumerate(models):
        m = model()
        print(f'Generating data for {type(m).__name__}')
        data = m.sample_n(15000)
        np.savetxt(f'data/{filenames[idx]}_train.csv',
                   data[:10000, :], delimiter=',')
        np.savetxt(f'data/{filenames[idx]}_test.csv',
                   data[10000:, :], delimiter=',')
