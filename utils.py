import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymysql
from scipy import stats, signal
import matplotlib.pyplot as plt
from scipy.stats import norm


class Alpha:
    name = ''
    region = ''
    T = None    # np.array of dates of size 1xT
    N = None    # np.array of stock ids of size 1xN
    alpha = None    # np.array of factor scores of size TxN
    holdings = None   # np.array of portfolio weights of size TxN

    def plot_moments(self):
        arr = zeroToNan(self.alpha)

        coverage = (~np.isnan(arr)).sum(axis=1)
        mini = np.nanmin(arr, axis=1)
        maxi = np.nanmax(arr, axis=1)
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        skew = stats.skew(arr, axis=1, nan_policy='omit')
        kurt = stats.kurtosis(arr, axis=1, nan_policy='omit')

        fig, ax = plt.subplots(5, 2, figsize=(23, 24))
        ax[0, 0].plot(self.T, coverage)
        ax[0, 0].grid()
        ax[0, 0].set_ylabel('Coverage')

        flat_arr = arr.flatten()
        flat_arr = flat_arr[(~np.isnan(flat_arr)) & (~np.isinf(flat_arr))]
        ax[0, 1].hist(flat_arr, bins=100)
        ax[0, 1].set_ylabel('Histogram')

        ax[1, 0].plot(self.T, mini)
        ax[1, 0].grid()
        ax[1, 0].set_ylabel('Minimum')

        ax[1, 1].plot(self.T, maxi)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel('Maximum')

        ax[2, 0].plot(self.T, mean)
        ax[2, 0].grid()
        ax[2, 0].set_ylabel('Mean')

        ax[2, 1].plot(self.T, std)
        ax[2, 1].grid()
        ax[2, 1].set_ylabel('std')

        ax[3, 0].plot(self.T, skew)
        ax[3, 0].grid()
        ax[3, 0].set_ylabel('skewness')

        ax[3, 1].plot(self.T, kurt)
        ax[3, 1].grid()
        ax[3, 1].set_ylabel('excess kurtosis')

        flat_arr = arr[:arr.shape[0] // 2, :].flatten()
        flat_arr = flat_arr[(~np.isnan(flat_arr)) & (~np.isinf(flat_arr))]
        ax[4, 0].hist(flat_arr, bins=100)
        ax[4, 0].set_ylabel('Histogram for first half of time sample')

        flat_arr = arr[arr.shape[0] // 2:, :].flatten()
        flat_arr = flat_arr[(~np.isnan(flat_arr)) & (~np.isinf(flat_arr))]
        ax[4, 1].hist(flat_arr, bins=100)
        ax[4, 1].set_ylabel('Histogram for second half of time sample')

        plt.show()

        lags = np.arange(20)
        ar = autocorrelation(arr, lags)
        plt.figure(figsize=(14, 4))
        plt.plot(lags, ar)
        plt.grid()
        plt.ylabel('Autocorrelation of factor scores')
        plt.xlabel('Lags')
        plt.xticks(lags)
        plt.show()
        print(f'Half-life is {np.argmax(np.array(ar) <= 0.5)}')


def get_df_from_db(query):
    db_connection_str = 'mysql+pymysql://root:evanescence-1@localhost/stocks'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(query, con=db_connection)
    db_connection.dispose()
    return df


def get_returns(N, T):
    df = get_df_from_db(f"""SELECT id, date, close FROM price_volume_US
                            JOIN mapping ON price_volume_US.ticker=mapping.ticker
                            WHERE price_volume_US.date >= {pd.to_datetime(T[0]).strftime('%Y%m%d')}
                            AND price_volume_US.date <= {pd.to_datetime(T[-1]).strftime('%Y%m%d')}
                            AND id in {'(' + ','.join([f"'{i}'" for i in N]) + ')'}""")
    df = df.pivot(index='date', columns='id', values='close')
    df = df.reindex(columns=N, index=T)
    close = np.array(df)
    ret = (close - shift(close, 1)) / shift(close, 1)
    return nanToZero(ret).clip(-0.3, 0.3)


def shift(arr, shift):
    """shift 2-D numpy array by amount shift along axis 0.
    Fill boundary values by nan."""
    out = np.zeros_like(arr)
    out[:] = np.nan
    if shift > 0:
        out[shift:, :] = arr[:-shift, :]
    elif shift < 0:
        out[:shift, :] = arr[-shift:, :]
    else:
        out = arr.copy()
    return out


def nanToZero(arr):
    out = arr.copy()
    out[np.isnan(out)] = 0
    return out


def zeroToNan(arr):
    out = arr.copy()
    out[out == 0] = np.nan
    return out


def zscore_cs(arr, floor=3):
    """compute cross-sectional zscore of a given numpy array. Results clipped in range [-floor, floor]."""
    mean = np.nanmean(arr, axis=1).reshape(-1,1)
    std = np.nanstd(arr, axis=1).reshape(-1,1)
    z = (arr - mean) / std
    return z.clip(-floor, floor)


def ewma(arr, hl, nhl=4):
    """take exponentially weighted moving average of arr across axis 0
    with halflife = hl and number of half lives = nhl"""
    assert not np.isnan(arr).any(), 'Remove NaN values before applying ewma'

    weights = np.array([2. ** (-t / hl) for t in np.arange(hl * nhl)]).reshape(-1, 1)

    out = signal.convolve(arr, weights, mode='full') / weights.sum()
    out = out[:-len(weights) + 1, :]
    return out


def normInv(arr):
    rank = np.array(pd.DataFrame(zeroToNan(arr)).rank(axis=1, pct=True))
    rank -= (np.nanmin(rank, axis=1) / 2).reshape(-1, 1)
    out =  norm.ppf(rank).clip(-3, 3)

    # due to equal values, sometimes cross-section mean can be different from 0
    out -= np.nanmean(out, axis=1).reshape(-1, 1)
    out /= np.nanstd(out, axis=1).reshape(-1, 1)
    return out


def autocorrelation(arr, lags=np.arange(10)):
    """Take care to not allow np.inf values in arr. These will lead to nan values."""
    in1 = zeroToNan(arr)
    ar = []
    for lag in lags:
        in2 = zeroToNan(shift(in1, lag))
        mask = ~(np.isnan(in1) | np.isnan(in2))
        ar.append(np.corrcoef(in1[mask], in2[mask])[0,1])
    return ar


def plot_autocorrelation(arr, lags=np.arange(10)):
    ar = autocorrelation(arr, lags)
    fig = plt.figure(figsize=(14,4))
    plt.plot(lags, ar)
    plt.grid()
    plt.ylabel('Autocorrelation')
    plt.xlabel('Lags')
    plt.xticks(lags)
    return fig


def ic_decay(arr, ret, lags=np.arange(-10,10), oos_idx=None, ts=False):
    """Take care to not allow np.inf values in arr and ret. These will lead to nan values."""
    if oos_idx:
        arr = arr.copy()
        arr[oos_idx, :] = np.nan

    in1 = zeroToNan(arr)
    in1 -= np.nanmean(in1, axis=1).reshape(-1, 1)
    in1 /= np.nanstd(in1, axis=1).reshape(-1, 1)

    in2 = zeroToNan(ret)
    in2 -= np.nanmean(in2, axis=1).reshape(-1, 1)
    in2 /= np.nanstd(in2, axis=1).reshape(-1, 1)

    ic = []
    for lag in lags:
        in1_lagged = shift(in1, lag)
        mask = ~(np.isnan(in1_lagged) | np.isnan(in2))
        ic.append(np.corrcoef(in1_lagged[mask], in2[mask])[0,1])

    if ts:
        # IC over time
        ic_ts = []
        for t in np.arange(in1.shape[0]):
            in11 = in1[t, :]
            in22 = in2[t, :]
            mask = ~(np.isnan(in11) | np.isnan(in22))
            ic_ts.append(np.corrcoef(in11[mask], in22[mask])[0, 1])

        print(f'Mean IC: {np.nanmean(ic_ts)}, std of IC: {np.nanstd(ic_ts)}')
        print(f'Ratio (annualised): {np.sqrt(252) * np.nanmean(ic_ts) / np.nanstd(ic_ts)}')

    return ic


def plot_ic_decay(arr, ret, lags=np.arange(-10,10), oos_idx=None, ts=False):
    ic = ic_decay(arr, ret, lags, oos_idx, ts)
    fig = plt.figure(figsize=(14,4))
    plt.plot(lags, ic)
    plt.grid()
    plt.ylabel('IC-decay')
    plt.xlabel('Lags')
    plt.xticks(lags)
    return fig


def plot_ic_vs_autocorrelation(arr_scores, arr_holdings, ret, lags=np.arange(10), oos_idx=None):
    ic = ic_decay(arr_holdings, ret, lags, oos_idx)
    ic /= ic[0]
    ar = autocorrelation(arr_scores, lags)

    fig = plt.figure(figsize=(14, 4))
    plt.plot(lags, ar, label='Autocorrelation')
    plt.plot(lags, ic, label='IC normalised')
    plt.grid()
    plt.legend()
    plt.xlabel('Lags')
    plt.xticks(lags)
    return fig


def momentum(data, hl_short, hl_long):
    fast = ewma(data, hl_short, 2)
    slow = ewma(data, hl_long, 2)
    diff = fast - slow
    diff /= diff.std(axis=1).reshape(-1,1)
    return diff.clip(-3,3)

