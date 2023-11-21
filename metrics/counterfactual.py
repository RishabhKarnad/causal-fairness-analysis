import numpy as np


# def ctf_de(y, x, scm):
#     prob = 0
#     for z in [0, 1]:
#         for w in [0, 1]:
#             prob += (
#                 ((scm.p_y_zxw(z, 1, w).pmf(y) - scm.p_y_zxw(z, 0, w).pmf(y)) * scm.p_w_zx(z, 0).pmf(w) * scm.p_x_z(z).pmf(x) * scm.p_z().pmf(z) / np.sum([scm.p_x_z(z_prime).pmf(x) * scm.p_z().pmf(z_prime) for z_prime in [0, 1]])))
#     return prob


def ctf_de(data, model, scm, x_0, x_1):
    m, n = data.shape

    zs, xs, ws = data[:, 0], data[:, 1], data[:, 2]

    ws_mc = scm.p_w_zx(None, x_0).rvs(m)
    zs_mc = scm.p_z().rvs(m)

    y_x1 = model.predict(
        np.vstack([zs_mc, np.ones(xs.shape)*x_1, ws_mc]).T).sum()
    y_x0 = model.predict(
        np.vstack([zs_mc, np.ones(xs.shape)*x_0, ws_mc]).T).sum()
    y_prob_x1 = model.predict_proba(
        np.vstack([zs_mc, np.ones(xs.shape)*x_1, ws_mc]).T).sum()
    y_prob_x0 = model.predict_proba(
        np.vstack([zs_mc, np.ones(xs.shape)*x_0, ws_mc]).T).sum()

    return (y_x1 - y_x0), (y_prob_x1 - y_prob_x0)


# def ctf_ie(y, x, scm):
#     prob = 0
#     for z in [0, 1]:
#         for w in [0, 1]:
#             prob += (
#                 (scm.p_y_zxw(z, 0, w).pmf(y) * (scm.p_w_zx(z, 1).pmf(w) - scm.p_w_zx(z, 0).pmf(w)) * scm.p_x_z(z).pmf(x) * scm.p_z().pmf(z) / np.sum([scm.p_x_z(z_prime).pmf(x) * scm.p_z().pmf(z_prime) for z_prime in [0, 1]])))
#     return prob

def ctf_ie(data, model, scm, x_0, x_1):
    m, n = data.shape

    zs, xs, ws = data[:, 0], data[:, 1], data[:, 2]

    ws_mc_x0 = scm.p_w_zx(None, x_0).rvs(m)
    ws_mc_x1 = scm.p_w_zx(None, x_1).rvs(m)
    zs_mc = scm.p_z().rvs(m)

    y_x1 = model.predict(
        np.vstack([zs_mc, np.ones(xs.shape)*x_1, ws_mc_x1]).T).sum()
    y_x0 = model.predict(
        np.vstack([zs_mc, np.ones(xs.shape)*x_0, ws_mc_x0]).T).sum()
    y_prob_x1 = model.predict_proba(
        np.vstack([zs_mc, np.ones(xs.shape)*x_1, ws_mc_x1]).T).sum()
    y_prob_x0 = model.predict_proba(
        np.vstack([zs_mc, np.ones(xs.shape)*x_0, ws_mc_x0]).T).sum()

    return (y_x1 - y_x0), (y_prob_x1 - y_prob_x0)


# def ctf_se(y, scm):
#     prob = 0
#     for z in [0, 1]:
#         for w in [0, 1]:
#             prob += (
#                 (scm.p_y_zxw(z, 0, w).pmf(y) * scm.p_w_zx(z, 0).pmf(w) * (
#                     scm.p_x_z(z).pmf(1) * scm.p_z().pmf(z) / np.sum(
#                         [scm.p_x_z(z_prime).pmf(1) * scm.p_z().pmf(z_prime) for z_prime in [0, 1]])
#                     - scm.p_x_z(z).pmf(0) * scm.p_z().pmf(z) / np.sum([scm.p_x_z(z_prime).pmf(0) * scm.p_z().pmf(z_prime) for z_prime in [0, 1]]))))
#     return prob

def ctf_se(data, model, scm, x_0, x_1):
    m, n = data.shape

    zs, xs, ws = data[:, 0], data[:, 1], data[:, 2]

    ws_mc = scm.p_w_zx(None, x_0).rvs(m)
    zs_mc_x0 = scm.p_z().rvs(m)
    zs_mc_x1 = scm.p_z().rvs(m)

    y_x1 = model.predict(
        np.vstack([zs_mc_x1, np.ones(xs.shape)*x_1, ws_mc]).T).sum()
    y_x0 = model.predict(
        np.vstack([zs_mc_x0, np.ones(xs.shape)*x_0, ws_mc]).T).sum()
    y_prob_x1 = model.predict_proba(
        np.vstack([zs_mc_x1, np.ones(xs.shape)*x_1, ws_mc]).T).sum()
    y_prob_x0 = model.predict_proba(
        np.vstack([zs_mc_x0, np.ones(xs.shape)*x_0, ws_mc]).T).sum()

    return (y_x1 - y_x0), (y_prob_x1 - y_prob_x0)
