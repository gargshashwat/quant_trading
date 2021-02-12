import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import warnings
warnings.filterwarnings('ignore')

from utils import *


def sample_covariance_matrix(N, T, end_date):
    """N is list of stock ids"""
    start_date = pd.to_datetime(end_date) - BDay(T)

    start = pd.to_datetime(start_date).strftime('%Y%m%d')
    end = pd.to_datetime(end_date).strftime('%Y%m%d')
    id_string = '(' + ','.join([f"'{i}'" for i in N]) + ')'

    df = get_df_from_db(f"""SELECT id, date, close FROM price_volume_US
                            JOIN mapping ON price_volume_US.ticker=mapping.ticker
                            WHERE price_volume_US.date >= {start}
                            AND price_volume_US.date <= {end}
                            AND id in {id_string}""")
    df = df.pivot(index='date', columns='id', values='close')
    df = df.reindex(columns=N)

    close = np.array(df)
    ret = (close - shift(close, 1)) / shift(close, 1)  # TxN matrix
    ret = nanToZero(ret)
    ret = ret.clip(-0.3, 0.3)

    m = ret.mean(axis=0)
    cov = (1 / ret.shape[0]) * np.dot((ret - m).T, ret - m)
    return cov


def smooth_noise_cov(cov, K=50):
    """make a new covariance matrix which keeps top K eigenvalues
     and makes other eigenvalues uniform in value by keeping trace as the invariant"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    diag = np.diag(eigenvalues)
    leftover_trace = np.trace(cov) - np.trace(diag[-K:, -K:])
    N = cov.shape[0]
    diag[np.arange(N - K), np.arange(N - K)] = leftover_trace / (N - K)
    return np.dot(eigenvectors, np.dot(diag, eigenvectors.T))


def cov_analysis(N, T, K):
    end_date = pd.to_datetime('2008-01-01')
    condition_numbers = {}
    max_eigenvalues = {}
    variance_aapl = {}
    variance_msft = {}
    corr_msft_aapl = {}
    while end_date < pd.to_datetime('2021-01-01'):
        cov = sample_covariance_matrix(N, T, end_date)
        cov = smooth_noise_cov(cov, K)
        eigenvalues = np.linalg.eigh(cov)[0]
        condition_numbers[end_date] = eigenvalues[-1] / eigenvalues[0]
        max_eigenvalues[end_date] = eigenvalues[-1]
        aapl_idx, msft_idx = list(N).index(9), list(N).index(1762)
        variance_aapl[end_date] = cov[aapl_idx, aapl_idx]
        variance_msft[end_date] = cov[msft_idx, msft_idx]
        corr_msft_aapl[end_date] = cov[aapl_idx, msft_idx] / np.sqrt(cov[aapl_idx, aapl_idx] * cov[msft_idx, msft_idx])
        end_date += BDay(20)

    plt.figure(figsize=(14, 4))
    plt.grid()
    plt.ylabel('Condition Number')
    plt.plot(np.array([i for i in condition_numbers.keys()]), np.array([i for i in condition_numbers.values()]))
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.grid()
    plt.ylabel('Max eigenvalue')
    plt.plot(np.array([i for i in max_eigenvalues.keys()]), np.array([i for i in max_eigenvalues.values()]))
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.grid()
    plt.ylabel('AAPL variance')
    plt.plot(np.array([i for i in variance_aapl.keys()]), np.array([i for i in variance_aapl.values()]))
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.grid()
    plt.ylabel('MSFT variance')
    plt.plot(np.array([i for i in variance_msft.keys()]), np.array([i for i in variance_msft.values()]))
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.grid()
    plt.ylabel('AAPL MSFT correlation')
    plt.plot(np.array([i for i in corr_msft_aapl.keys()]), np.array([i for i in corr_msft_aapl.values()]))
    plt.show()


# T = 250 gives lowest rms error
def cov_rmse(N, T, K):
    rmses = []
    end_date = pd.to_datetime('2010-01-01')
    while end_date < pd.to_datetime('2021-01-01'):
        print(end_date)
        cov = sample_covariance_matrix(N, T, end_date)
        cov = smooth_noise_cov(cov, K)

        end_date += BDay(20)
        cov_next = sample_covariance_matrix(N, 20, end_date)

        rms = np.mean((cov - cov_next) ** 2)
        rmses.append(rms)
    return np.mean(rmses)


def save_cov_inv():
    """Save inverse covariance matrices for a fixed subset of stock and dates
    make reserach faster."""
    N = get_df_from_db(f"""SELECT * FROM mapping""").id.tolist()[:500]
    for end_date in pd.date_range('2015-01-01', '2021-01-01', freq='MS'):
        cov = sample_covariance_matrix(N, 250, end_date)
        cov = smooth_noise_cov(cov, K=10)
        cov_inv = np.linalg.inv(cov)
        np.save(f"inv_cov/{pd.to_datetime(end_date).strftime('%Y%m%d')}.npy", cov_inv)