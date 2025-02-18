import pandas as pd
import numpy as np
import time


from utils import *


class Reversion(Alpha):
    def get(self, start, end):
        # load pricing data
        t1 = time.time()
        df = get_df_from_db(f"""SELECT id, date, close FROM price_volume_US
                            JOIN mapping ON price_volume_US.ticker=mapping.ticker
                            WHERE price_volume_US.date >= {pd.to_datetime(start).strftime('%Y%m%d')}
                            AND price_volume_US.date <= {pd.to_datetime(end).strftime('%Y%m%d')}""")
        print(f'Time taken to retrieve data: {time.time()-t1}s')

        # pivot
        df = df.pivot(index='date', columns='id', values='close')

        # convert to numpy array
        close = np.array(df)

        alpha = Alpha()
        alpha.N = df.columns
        alpha.T = df.index
        alpha.region = 'US'
        alpha.name = 'reversion'
        alpha.alpha = - (close - shift(close, 1)) / shift(close, 1)
        alpha.alpha = shift(alpha.alpha, 1)  # delay alpha by one day

        alpha.alpha = alpha.alpha.clip(-0.1, 0.1)
        alpha.alpha = ewma(nanToZero(alpha.alpha), 5, 2)
        alpha.alpha = normInv(alpha.alpha)
        return alpha


class Momentum(Alpha):
    def get(self, start, end):
        df = get_df_from_db(f"""SELECT id, date, close FROM price_volume_US
                            JOIN mapping ON price_volume_US.ticker=mapping.ticker
                            WHERE price_volume_US.date >= {pd.to_datetime(start).strftime('%Y%m%d')}
                            AND price_volume_US.date <= {pd.to_datetime(end).strftime('%Y%m%d')}""")

        # pivot
        df = df.pivot(index='date', columns='id', values='close')

        # fill nan values
        df = df.fillna(method='ffill', axis=0).fillna(0)

        # convert to numpy array
        close = np.array(df)

        # make alpha
        mom_close = momentum(close, 5, 20)

        alpha = Alpha()
        alpha.N = df.columns
        alpha.T = df.index
        alpha.region = 'US'
        alpha.name = 'momentum'
        alpha.alpha = shift(mom_close, 1)  # delay alpha by one day

        return alpha
