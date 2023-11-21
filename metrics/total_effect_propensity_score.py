import numpy as np
from causalinference import CausalModel


def estimate_ate(data, model):
    y = model.predict(data[:, :3])
    cm = CausalModel(y, data[:, 1], data[:, (0, 2)])
    cm.est_via_matching(matches=1, bias_adj=True)
    return cm.estimates['matching']['ate']


def estimate_ett(data, model, x_0, x_1):
    # data_filtered = data[(data[:, 1] <= x_0+0.25 and data[:, 1] >= x_0-0.25)]
    x_mean = (x_0 + x_1) / 2
    data[:, 1] = np.where(data[:, 1] < x_mean, 0, 1)
    cate = estimate_ate(data, model)
    return cate
