import numpy as np

from data.binary import FairSCM, UnfairSCM
from data.linear import FairGaussianSCM, UnfairGaussianSCM
from data.nonlinear import FairNonlinearGaussianSCM, UnfairNonlinearGaussianSCM

from metrics.total_variation import tv
from metrics.counterfactual import ctf_de, ctf_ie, ctf_se


def evaluate(data, model):
    print(f'Evaluating {type(model).__name__}')

    tv_scores = tv(data[:, -1], model)
    de_scores = ctf_de(data[:, -1], data[:, 1], model)
    ie_scores = ctf_ie(data[:, -1], data[:, 1], model)
    se_scores = ctf_se(data[:, -1], model)

    max_discriminated_idx = np.argmax(tv_scores)

    print(f'TV: {tv_scores[max_discriminated_idx]}')
    print(
        f'DE - IE - SE: {-de_scores[max_discriminated_idx] + ie_scores[max_discriminated_idx] + se_scores[max_discriminated_idx]}')
    print()


def main():
    models = [
        (FairSCM().sample_n(100), FairSCM()),
        (UnfairSCM().sample_n(100), UnfairSCM()),
    ]
    [evaluate(data, model) for (data, model) in models]


if __name__ == '__main__':
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1]:
        print(alpha)
        print(FairGaussianSCM(alpha).sample_n(10))
        print(UnfairGaussianSCM(alpha).sample_n(10))
        print(FairNonlinearGaussianSCM(alpha).sample_n(10))
        print(UnfairNonlinearGaussianSCM(alpha).sample_n(10))
        print('===============')

    main()