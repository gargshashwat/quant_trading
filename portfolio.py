import numpy as np
import multiprocessing as mp

from covariance import *
from utils import *


def sub_cp(t):
    (end_date, sub_alpha, N, mode) = t

    if mode == 'research':
        with open(f"inv_cov/{pd.datetime(end_date.year, end_date.month, 1).strftime('%Y%m%d')}.npy", 'rb') as f:
            cov_inv = np.load(f)
        # restrict to those ids which are in N. These must be a subset of [1,..,500]
        idx = [i - 1 for i in N]
        cov_inv = cov_inv[np.ix_(idx, idx)]
    else:
        cov = sample_covariance_matrix(N, 250, pd.datetime(end_date.year, end_date.month, 1))
        cov = smooth_noise_cov(cov, K=10)
        cov_inv = np.linalg.inv(cov)

    w = np.dot(nanToZero(cov_inv), nanToZero(sub_alpha.T))
    return w.T


def characteristic_portfolio(alpha, mode):
    start_dates = [(i, d) for i, d in enumerate(alpha.T) if d.month != alpha.T[i - 1].month]
    end_dates = [(i - 1, alpha.T[i - 1]) for (i, d) in start_dates[1:]] + [(len(alpha.T) - 1, alpha.T[-1])]
    date_ranges = [(i, j, s, e) for ((i, s), (j, e)) in zip(start_dates, end_dates)]
    mp_input = [(e, alpha.alpha[i:j+1, :], alpha.N, mode) for (i, j, s, e) in date_ranges]

    pool = mp.Pool(processes=4)
    results = pool.map(sub_cp, mp_input)
    pool.close()
    w =  nanToZero(np.concatenate(results, axis=0))

    # trim outliers
    nn = np.percentile(w, [99], axis=1).T
    w = np.where(w <= nn, w, nn)
    nn = np.percentile(w, [1], axis=1).T
    w = np.where(w >= nn, w, nn)

    # remove micro-weights and normalise to 1 everyday
    w2 = (w.T / np.sum(np.abs(w), axis=1)).T
    w = np.where(np.abs(w2) > 1e-5, w2, 0)

    # normalise to 1 million everyday
    w *= 1e6

    return zeroToNan(w)