import numpy as np
import joblib

import warnings

from data.binary import FairSCM, UnfairSCM
from data.linear import FairGaussianSCM, UnfairGaussianSCM
from data.nonlinear import FairNonlinearGaussianSCM, UnfairNonlinearGaussianSCM

from metrics.total_variation import tv
from metrics.counterfactual import ctf_de, ctf_ie, ctf_se
from metrics.total_effect_propensity_score import estimate_ett

from data.generate import generate


warnings.simplefilter('ignore')

# def evaluate(data, model):
#     print(f'Evaluating {type(model).__name__}')

#     tv_scores = tv(data[:, -1], model)
#     de_scores = ctf_de(data[:, -1], data[:, 1], model)
#     ie_scores = ctf_ie(data[:, -1], data[:, 1], model)
#     se_scores = ctf_se(data[:, -1], model)

#     max_discriminated_idx = np.argmax(tv_scores)

#     print(f'TV: {tv_scores[max_discriminated_idx]}')
#     print(
#         f'DE - IE - SE: {-de_scores[max_discriminated_idx] + ie_scores[max_discriminated_idx] + se_scores[max_discriminated_idx]}')
#     print()


# def main():
#     models = [
#         (FairSCM().sample_n(100), FairSCM()),
#         (UnfairSCM().sample_n(100), UnfairSCM()),
#     ]
#     [evaluate(data, model) for (data, model) in models]


def evaluate():
    scms = ['fair', 'unfair']
    model_classes = ['linear', 'nonlinear']
    models = ['dt', 'LR', 'mlp']
    scm_classes = {
        'fair': {
            'linear': FairGaussianSCM,
            'nonlinear': FairNonlinearGaussianSCM,
        },
        'unfair': {
            'linear': UnfairGaussianSCM,
            'nonlinear': UnfairNonlinearGaussianSCM,
        },
    }

    for scm in scms:
        for model in models:
            scm_obj = scm_classes[scm]['linear']()
            path = f'trained_models/{scm}/linear/{model}.joblib'
            data = np.genfromtxt(f'data/{scm}_linear_test.csv', delimiter=',')
            clf = joblib.load(path)
            x0, x1 = data[:, 1].mean(), data[:, 1].mean()+data[:, 1].std()
            print(f'trained_models/{scm}/linear/{model}')
            print(ctf_de(data, clf, scm_obj, x0, x1))
            print(ctf_ie(data, clf, scm_obj, x0, x1))
            print(ctf_se(data, clf, scm_obj, x0, x1))
            print(estimate_ett(data, clf, x0, x1))
            print('==========================')

    for scm in scms:
        for model in models:
            scm_obj = scm_classes[scm]['nonlinear']()
            path = f'trained_models/{scm}/nonlinear/{model}.joblib'
            data = np.genfromtxt(
                f'data/{scm}_nonlinear_test.csv', delimiter=',')
            clf = joblib.load(path)
            x0, x1 = data[:, 1].mean(), data[:, 1].mean()+data[:, 1].std()
            print(f'trained_models/{scm}/nonlinear/{model}')
            print(ctf_de(data, clf, scm_obj, x0, x1))
            print(ctf_ie(data, clf, scm_obj, x0, x1))
            print(ctf_se(data, clf, scm_obj, x0, x1))
            print(estimate_ett(data, clf, x0, x1))
            print('==========================')


if __name__ == '__main__':
    # print(FairGaussianSCM().sample_n(10))
    # print(UnfairGaussianSCM().sample_n(10))
    # print(FairNonlinearGaussianSCM().sample_n(10))
    # print(UnfairNonlinearGaussianSCM().sample_n(10))
    # print('===============')

    # generate()

    evaluate()
