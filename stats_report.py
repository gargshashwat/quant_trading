import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils import *
from portfolio import *
from data_to_db import *


def calculate_pnl(w, ret, T, lags=np.arange(21)):
    """using close to close returns: this is not realistic but gives a good approximation"""
    pnls = {lag: nanToZero(shift(w, lag) * ret).sum(axis=1) for lag in lags}

    fig = plt.figure(figsize=(14, 4))
    for lag in [0, 1, 2, 5, 10, 20]:
        cumpnl = np.cumsum(pnls[lag])
        cumpnl = np.where(np.isnan(w).all(axis=1), np.nan, cumpnl)
        plt.plot(T, cumpnl, label=f'Lag {lag}')
    plt.grid()
    plt.legend()
    plt.title('PnL')
    return pnls, fig


def drawdown(pnl, w, T):
    cumpnl = np.cumsum(pnl)
    mx = np.maximum.accumulate(cumpnl)
    dd = cumpnl - mx
    dd = np.where(np.isnan(w).all(axis=1), np.nan, dd)

    fig = plt.figure(figsize=(14, 4))
    plt.plot(T, dd)
    plt.grid()
    plt.ylabel('Drawdown')
    return fig


def calculate_ir(pnls):
    irs = []
    se_ir = []
    for key, pnl in pnls.items():
        pnl2 = zeroToNan(pnl)
        ir = (np.nanmean(pnl2) / np.nanstd(pnl2)) * np.sqrt(252)
        irs.append(ir)
        se_ir.append(np.sqrt((1 + 0.5 * ir ** 2) / (np.isfinite(pnl2).sum() / 252)))

    fig = plt.figure(figsize=(9, 6))
    plt.plot(list(pnls.keys()), irs)
    plt.fill_between(list(pnls.keys()), np.array(irs) - np.array(se_ir), np.array(irs) + np.array(se_ir), alpha=0.3)
    plt.grid()
    plt.ylabel('IR')
    plt.xlabel('Lag')
    return irs, fig


def ir_decile(w, ret):
    """make deciles as per values of w, not alpha, and calculate alpha for each decile"""
    irs = []
    se_ir = []
    for p in np.arange(0, 100, 10):
        p_lower = np.nanpercentile(w, [p], axis=1).T
        p_upper = np.nanpercentile(w, [p + 10], axis=1).T
        w_p = np.where((w >= p_lower) & (w < p_upper), w, np.nan)
        pnl = zeroToNan(nanToZero(w_p * ret).sum(axis=1))

        ir = np.nanmean(pnl) / np.nanstd(pnl) * np.sqrt(252)
        irs.append(ir)
        se_ir.append(np.sqrt((1 + 0.5 * ir ** 2) / (np.isfinite(pnl).sum() / 252)))

    fig = plt.figure(figsize=(9, 6))
    plt.errorbar(np.arange(1, 11), irs, yerr=se_ir, fmt='s')
    plt.grid()
    plt.xticks(np.arange(1, 11))
    plt.ylabel('IR')
    plt.xlabel('Decile')
    return fig


def calculate_stats(w, ret, pnls):
    turnover = np.abs(nanToZero(w) - nanToZero(shift(w, 1))).sum(axis=1) / 1e4
    turnover = np.nanmean(zeroToNan(turnover))

    hp = 200 / turnover

    pnl = zeroToNan(pnls[0])

    roe = np.nanmean(pnl) * 252 / 1e4

    rot = np.nanmean(pnl) * 100 / (turnover * 1e4)

    trades_won = (nanToZero(w * ret) > 0).sum()
    trades_lost = (nanToZero(w * ret) < 0).sum()
    hit_ratio = trades_won / (trades_won + trades_lost)

    each_pnl = nanToZero(w * ret)
    avg_profit = each_pnl[each_pnl > 0].mean()
    avg_loss = - each_pnl[each_pnl < 0].mean()
    PvsL_ratio = avg_profit / avg_loss

    irs = [(np.nanmean(pnl) / np.nanstd(pnl)) * np.sqrt(252) for pnl in pnls.values()]
    irs /= irs[0]
    halflife = np.argmax(irs <= 0.5)

    ir = (np.nanmean(pnl) / np.nanstd(pnl)) * np.sqrt(252)

    df = pd.DataFrame({'IR': round(ir, 2),
                       'turnover %': round(turnover, 0),
                       'HP': round(hp, 1),
                       'ROE (%)': round(roe, 2),
                       'ROT (bps)': round(100 * rot, 2),
                       'hit_ratio %': round(100 * hit_ratio, 1),
                       'P/L': round(PvsL_ratio, 3),
                       'halflife': halflife}, index=[0])
    return df


def ir_up_down_markets(pnl, T):
    gspc = get_df_from_db("SELECT * FROM sp500_US")
    gspc['close_20ma'] = gspc['close'].rolling(20).mean()
    gspc['close_100ma'] = gspc['close'].rolling(100).mean()

    bear_runs = gspc.date[gspc.close_20ma < gspc.close_100ma]
    bear_runs = [i for i in bear_runs.index if ((i - 1) not in bear_runs.index) or ((i + 1) not in bear_runs.index)]
    bear_runs = [(gspc.date[bear_runs[i]], gspc.date[bear_runs[i + 1]]) for i in np.arange(0, len(bear_runs), 2)
                 if bear_runs[i + 1] - bear_runs[i] >= 40]
    bear_ranges = [pd.date_range(i, j) for (i, j) in bear_runs]
    bear_dates = [i.date() for li in bear_ranges for i in li]

    bear_idx = [idx for idx, d in enumerate(T) if d in bear_dates]
    bull_idx = [i for i in np.arange(len(T)) if i not in bear_idx]

    pnl_bear = zeroToNan(pnl[bear_idx])
    ir_bear = np.nanmean(pnl_bear) / np.nanstd(pnl_bear) * np.sqrt(252)
    se_ir_bear = np.sqrt((1 + 0.5 * ir_bear ** 2) / (np.isfinite(pnl_bear).sum() / 252))

    pnl_bull = zeroToNan(pnl[bull_idx])
    ir_bull = np.nanmean(pnl_bull) / np.nanstd(pnl_bull) * np.sqrt(252)
    se_ir_bull = np.sqrt((1 + 0.5 * ir_bull ** 2) / (np.isfinite(pnl_bull).sum() / 252))

    fig = plt.figure(figsize=(6, 4))
    plt.errorbar(['Bear', 'Bull'], [ir_bear, ir_bull], yerr=[se_ir_bear, se_ir_bull], fmt='s')
    plt.grid()
    plt.title('IR during up and down markets')
    return fig


def ir_vix_quantiles(pnl, T):
    vix = get_df_from_db("SELECT * FROM vix_US")
    vix_quantiles = list(vix.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).close)

    irs = []
    se_irs = []
    for i in [0, 1, 2, 3, 4]:
        mask = (vix.close >= vix_quantiles[i]) & (vix.close < vix_quantiles[i + 1])
        vix_idx = [idx for idx, d in enumerate(T) if d in list(vix[mask].date)]

        pnl_vix = zeroToNan(pnl[vix_idx])
        ir = np.nanmean(pnl_vix) / np.nanstd(pnl_vix) * np.sqrt(252)
        se_ir = np.sqrt((1 + 0.5 * ir ** 2) / (np.isfinite(pnl_vix).sum() / 252))
        irs.append(ir)
        se_irs.append(se_ir)

    fig = plt.figure(figsize=(6, 4))
    plt.errorbar([1, 2, 3, 4, 5], irs, yerr=se_irs, fmt='s')
    plt.grid()
    plt.xticks([1, 2, 3, 4, 5])
    plt.title('IR in different VIX quantiles')
    return fig


def generate_alpha_report(alpha, mode='prod', dollar_neutral=False, hide_oos=True):
    alpha.holdings = characteristic_portfolio(alpha, mode, dollar_neutral=dollar_neutral)

    oos_idx = None
    if hide_oos:
        oos_idx = [idx for idx, d in enumerate(alpha.T) if
                         (d.year % 2 == 0 and np.ceil(d.month / 3) % 2 == 0) or (d.year % 2 == 1 and np.ceil(d.month / 3) % 2 == 1)]
        alpha.holdings[oos_idx, :] = np.nan

    ret = get_returns(alpha.N, alpha.T)
    pnls, fig1 = calculate_pnl(alpha.holdings, ret, alpha.T)
    fig2 = drawdown(pnls[0], alpha.holdings, alpha.T)
    irs, fig3 = calculate_ir(pnls)
    fig4 = ir_decile(alpha.holdings, ret)
    df = calculate_stats(alpha.holdings, ret, pnls)
    fig5 = ir_up_down_markets(pnls[0], alpha.T)
    fig6 = ir_vix_quantiles(pnls[0], alpha.T)
    fig7 = plot_autocorrelation(alpha.alpha)

    fig8 = plot_ic_decay(alpha.holdings, ret, oos_idx=oos_idx)
    fig9 = plot_ic_decay(alpha.holdings, ret, oos_idx=oos_idx, lags=np.arange(10), ts=True)
    fig10 = plot_ic_vs_autocorrelation(alpha.alpha, alpha.holdings, ret, oos_idx=oos_idx)

    # distribution of cumpnl per stock
    each_pnl = nanToZero(alpha.holdings * ret)
    cumpnl = np.cumsum(each_pnl, axis=0)
    fig11 = plt.figure(figsize=(14, 4))
    plt.hist(cumpnl[-1, :], bins=100)
    plt.grid()

    pp = PdfPages(f'reports/{alpha.name}_{alpha.region}.pdf')

    fig = plt.figure(figsize=(9, 2))
    c = df.shape[1]
    plt.table(cellText=np.vstack([df.columns, df.values]), cellColours=[['lightgray'] * c] + [['none'] * c], bbox=[0, 0, 1, 1])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False, labelright=False)
    pp.savefig(fig)

    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.savefig(fig4)
    pp.savefig(fig5)
    pp.savefig(fig6)
    pp.savefig(fig7)
    pp.savefig(fig8)
    pp.savefig(fig9)
    pp.savefig(fig10)
    pp.savefig(fig11)
    pp.close()


def mini_alpha_report(alpha, ret, mode='research', dollar_neutral=False, hide_oos=True):
    alpha.holdings = characteristic_portfolio(alpha, mode, dollar_neutral=dollar_neutral)

    if hide_oos:
        oos_idx = [idx for idx, d in enumerate(alpha.T) if
                         (d.year % 2 == 0 and np.ceil(d.month / 3) % 2 == 0) or (d.year % 2 == 1 and np.ceil(d.month / 3) % 2 == 1)]
        alpha.holdings[oos_idx, :] = np.nan

    turnover = np.abs(nanToZero(alpha.holdings) - nanToZero(shift(alpha.holdings, 1))).sum(axis=1) / 1e4
    turnover = zeroToNan(turnover)
    if hide_oos:
        turnover[oos_idx] = np.nan
    turnover = np.nanmean(turnover)

    pnl = zeroToNan(nanToZero(alpha.holdings * ret).sum(axis=1))
    roe = np.nanmean(pnl) * 252 / 1e4
    rot = np.nanmean(pnl) * 100 / (turnover * 1e4)

    trades_won = (nanToZero(alpha.holdings * ret) > 0).sum()
    trades_lost = (nanToZero(alpha.holdings * ret) < 0).sum()
    hit_ratio = trades_won / (trades_won + trades_lost)

    each_pnl = nanToZero(alpha.holdings * ret)
    avg_profit = each_pnl[each_pnl > 0].mean()
    avg_loss = - each_pnl[each_pnl < 0].mean()
    PvsL_ratio = avg_profit / avg_loss

    ir = (np.nanmean(pnl) / np.nanstd(pnl)) * np.sqrt(252)

    df = pd.DataFrame({'IR': round(ir, 2),
                       'turnover %': round(turnover, 0),
                       'ROE (%)': round(roe, 2),
                       'ROT (bps)': round(100 * rot, 2),
                       'hit_ratio %': round(100 * hit_ratio, 1),
                       'P/L': PvsL_ratio}, index=[0])

    display(df)
