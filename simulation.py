import pandas as pd


import numpy as np
from datetime import datetime as dt, timedelta
from database import Connect
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# internal
from util import Dates, Misc, Returns

class SimParams():

    def __init__(self, start_date, end_date, sector=None, capital=None, weighting=None, reb_freq=None):
        self.start_date = start_date
        self.end_date = end_date
        self.sector = sector
        self.capital = capital
        self.weighting = weighting
        self.reb_freq = reb_freq
        
class DataParams(SimParams):

    def __init__(self, start_date, end_date, sector=None, capital=None, weighting=None, reb_freq=None):
        super().__init__(start_date, end_date, sector, capital, weighting, reb_freq)
        
        if self.weighting == 'VOL':
            # lookback period for vol-calculation
            self.lookback = 63
            self.data_start_date = Dates().add_business_days(date_ = self.start_date, days = -self.lookback)
        else:
            self.data_start_date = start_date

class ImportData(DataParams):

    def __init__(self, start_date, end_date, sector=None, capital=None, weighting=None, reb_freq=None):
        super().__init__(start_date, end_date, sector, capital, weighting, reb_freq)

    def __conform_price_data(self, price_df):
        '''
            conforming price_data to weekday-format and imputing missing values
        '''
        ticker_list = list(set(price_df['yh_id']))
        weekday = Dates().create_weekday_template(start_date=self.data_start_date, end_date=self.end_date)

        # subset df on symbol and merge on weekdays
        odf = pd.DataFrame()
        for t in ticker_list:
            df_sub = price_df.loc[price_df['yh_id'] == t].copy() 
            df_sub = weekday.merge(df_sub, on='date_', how='left')
            df_sub['yh_id'].fillna(method='bfill', inplace=True)
            df_sub['yh_id'].fillna(method='ffill', inplace=True)
            df_sub['price'].fillna(method='ffill', inplace=True)

            odf = odf.append(df_sub)

        odf.reset_index(drop=True, inplace=True)

        return odf

    def __vectorize_tickers(self, price_df):
            '''
            '''
            tickers = np.array(price_df['yh_id'])
            # unique tickers - keeping order
            _, idx = np.unique(tickers, return_index=True)
            tickers_unique = np.take(tickers, np.sort(idx))

            return np.sort(tickers_unique)

    def __vectorize_dates(self, price_df):

        dates = np.array(price_df['date_'])

        return np.unique(dates)
    
    def __vectorize_prices(self, price_df, tickers, dates):
        '''
            from price_df.col to (mxn) np.array 
        '''
        # (no of assets) x (no of unique dates)
        prices = np.zeros((len(tickers), len(dates)))
        for idx, ticker in enumerate(tickers):
            df_sub = price_df.loc[price_df['yh_id']==ticker].copy()
            prices[idx] = df_sub['price']
            
        return prices
    
    def __vectorize_returns(self, price_arr):
        
        ret = Returns()
        return ret.impute_with_mean(ret.compute_d_returns(price_arr))

    def __import_sector_constituents(self):
        '''
            importing s&p500 const from db

            out:
                <List>
        '''
        db = Connect('homedev')
        if self.sector == 'ALL':
            df = db.import_from_db(
                table='sp500_constituents',
                columns='*',
                )
        else:
            df = db.import_from_db(
                table='sp500_constituents',
                columns='*',
                condition = f"where sector = '{self.sector}'"
                )

        return df['yh_id'].to_list()
        
    def import_price_data(self, ticker_list=None):
        '''
            importing price data from database, imputing and conforming to weekday-standard (no date holes)

            out:
            <DataFrame>
                date_, yh_id, price
        '''

        if ticker_list:
            ticker_sql = Misc().list_to_sql(ticker_list)
        else:
            ticker_sql = Misc().list_to_sql(self.__import_sector_constituents())
        
        db = Connect('homedev')
        df = db.import_from_db(
            table='d_price',
            columns='*',
            condition=f"where yh_id in ({ticker_sql}) and date_ >= '{self.data_start_date}' and date_ <='{self.end_date}'"
            )

        # format date_ col
        df = Dates().str_to_date(df, 'date_')

        df = self.__conform_price_data(df)
        
        return df

    def vectorize_data(self, df):
        '''
            tickers     <np.array>  (1x1) 
            date_       <np.array>  (1x1)
            prices      <np.array>  (tickers x date_)
            returns     <np.array>  (tickers x date_)
        '''
        tickers = self.__vectorize_tickers(price_df=df)
        data_dates = self.__vectorize_dates(price_df=df)
        prices = self.__vectorize_prices(price_df=df, tickers=tickers, dates=data_dates)
        returns = self.__vectorize_returns(price_arr=prices)

        return {'tickers': tickers, 'data_dates': data_dates, 'returns': returns}
  

class Simulation(ImportData):
    
    def __init__(self, start_date, end_date, sector, capital, weighting, reb_freq):
        super().__init__(start_date, end_date, sector, capital, weighting, reb_freq)
        
        data_ = self.vectorize_data(self.import_price_data())

        self.sim_dates = np.array(Dates().create_weekday_template(self.start_date, self.end_date)['date_'])
        self.tickers = data_['tickers']

        if self.weighting == 'VOL':
            # extended return array due to lookback for rolling vol computation.
            # Workaroung...not great...not terrible
            self.returns_ext = data_['returns']
            self.returns = self.returns_ext[0:,self.lookback:]
            self.dates_ext = data_['data_dates']
        else:
            self.returns_ext = None
            self.returns = data_['returns']
            self.data_dates = self.sim_dates

    def __subset_array(self, arr, from_date, to_date=None):
        '''
            subsets array on dates
        '''
        if to_date is not None:
            idx = np.where((self.sim_dates >= from_date) & (self.sim_dates <= to_date))
        else:
            idx = np.where(self.sim_dates >= from_date)

        # more-dim-array
        if len(arr.shape) > 1:
            arr_sub = np.zeros((arr.shape[0], len(idx)))

            for i in range(arr.shape[0]):
                if i == 0:
                    arr_sub = np.array([arr[i][idx]])
                else:
                    arr_sub = np.r_[arr_sub, [arr[i][idx].T]]

            return arr_sub

        # one-dim-array
        else:
            return arr[idx]

    def __create_trade_dates(self):
        '''
        out:
            np.array(dt.timestamps)
        '''
        df = Dates().last_bday_in_month(self.start_date, self.end_date)
        if self.start_date not in df['date_']:
            add = pd.DataFrame({'date_': dt.strptime(self.start_date, '%Y-%m-%d')}, index=[0])
            df = df.append(add)
            
        trade_dates = [d for d in df['date_']]

        if self.reb_freq == 'M':
            return sorted(trade_dates)

        if self.reb_freq == 'Q':
            return sorted(trade_dates)[0::3]

        if self.reb_freq == 'S':
            return sorted(trade_dates)[0::6]
        
        if self.reb_freq == 'Y':
            return sorted(trade_dates)[0::12]
        
    def __compute_rolling_vols(self):
        '''
            computing rolling ann.vol

            return mxn np.array with no nan´s (i.e. return-shape equals sim-date-period with no lookback-period)
        '''        
        vol_ext = np.zeros((len(self.tickers), len(self.dates_ext)))
        vol_sim =  np.zeros((len(self.tickers), len(self.sim_dates)))

        for asset in range(len(self.tickers)):
            vol_ext[asset] = (np.array(pd.Series(self.returns_ext[asset]).rolling(self.lookback).std()))*(252**0.5)
            vol_sim[asset] = vol_ext[asset][self.lookback:]
                
        return vol_sim
    
    def __create_target_weights(self):
        ''' 
            target weights 
        '''

        # weights = np.ones(self.prices.shape)
        weights = np.ones(self.returns.shape)
        
        # equal weight
        if self.weighting == 'EW':

            weights = weights / len(self.tickers)

        # inverse vol weighting and normalizing
        if self.weighting == 'VOL':
            weights = weights / self.__compute_rolling_vols()
            weights = (weights / weights.sum(axis=0))
            
        return weights
    
    def simulate(self):
        '''
            iterating through trade-date-periods and simualting performance
        ''' 
        trade_dates = self.__create_trade_dates()
        for i, from_date in enumerate(trade_dates):

            if i < len(trade_dates)-1:
                to_date = trade_dates[i+1]
            else:
                # last trade date, i.e. no end_date
                to_date = None
            
            # SUBSET PRICE DATA ON TIME-PERIOD / COMPUTE TR-INDEX PER ASSET 
            returns_sub = self.__subset_array(arr=self.returns, from_date=from_date, to_date=to_date)
            ret = Returns()
            tr_idx = ret.compute_total_return_idx(returns_sub)            

            # DRIFT MV PER-ASSET WITH NEW WEIGHTS
            target_weights = self.__create_target_weights()
            if i == 0:
                nav = self.capital
                # select target-allocation on rebalancing date (i.e. from_date)
                target_mv = nav * self.__subset_array(arr=target_weights, from_date=from_date, to_date=from_date)
                target_mv = target_mv.squeeze()
                mv_assets = (target_mv * tr_idx.T).T
                # flow_assets = np.zeros(mv_assets.shape)
            else:
                # nav on last available date)                
                nav = sum([asset[-1] for asset in mv_assets])

                # select target-allocation on rebalancing date (i.e. from_date)
                target_mv = nav * self.__subset_array(arr=target_weights, from_date=from_date, to_date=from_date)
                target_mv = target_mv.squeeze()
                mv_drift = (target_mv * tr_idx.T).T
                mv_assets = np.append(mv_assets, mv_drift[0:,1:], axis=1)
                
                # # COMPUTING TRADE FLOWS
                # # trade_flow_new = mv (after trade) - mv (before trade)
                # flow_assets_new = np.zeros(mv_drift.shape)
                # flow_assets_new[0:,1] = target_mv - mv_assets[0:,-1]

                # flow_assets = np.append(flow_assets, flow_assets_new[0:,1:], axis=1)
        
        return mv_assets


class AnalyseSim():

    def __init__(self, dates, tickers, asset_d_mv, asset_d_returns):    
        
        self.dates = dates
        self.tickers = tickers
        self.asset_d_mv = asset_d_mv
        self.asset_d_returns = asset_d_returns
        self.pf_tr = self.__pf_tr()

    def __pf_nav(self):
        '''
            aggregates asset_mv´s
        '''
        return np.sum(self.asset_d_mv, axis=0)

    def __pf_d_returns(self):
        '''
            computing pf daily-returns
        '''
        return Returns().compute_d_returns(self.__pf_nav())        

    def __pf_tr(self):
        '''
            computing pf total-return-index
        '''
        pf_tr = Returns().compute_total_return_idx(self.__pf_d_returns())

        return pf_tr

    def __asset_weights(self):
        '''
            computing asset weights per date
        '''

        return self.asset_d_mv / self.__pf_nav()

    def max_dd(self):

        peaks = pd.Series(self.pf_tr).cummax()
        drawdown = (pd.Series(self.pf_tr) - peaks) / peaks
        max_dd = min(drawdown)

        return max_dd

    def pf_metrics(self):
        '''
            annual_ret
            annual_vol
            information_ratio
            dd
        '''


        anu_ret = self.pf_tr[-1]**(252/len(self.pf_tr)) -1
        anu_vol = np.std(self.__pf_d_returns())*252 ** 0.5
        ir = anu_ret/anu_vol
        dd = self.max_dd()
        no_of_assets = len(self.asset_d_mv)

        return anu_ret, anu_vol, ir, dd, no_of_assets

    def pf_tr_to_df(self):
        '''
            pf_tr to df 
        '''
        df = pd.DataFrame(
            {
                'date_': self.dates,
                'pf_tr': self.pf_tr
            }
        )

        return df

    def carino_perf_contribution(self):
        '''
            computing performance-contribution with carino smoothing.
            (sum of asset contributions add up to total pf-return) 
        '''
        contr = self.__asset_weights() * self.asset_d_returns
        contr_t = np.sum(contr, axis=0)
        
        pf_ret = np.cumprod(1+contr_t) -1

        k = np.log(1+pf_ret[-1])/pf_ret[-1]

        k_t = np.zeros(len(contr_t))
        for i in range(len(contr_t)):
            if contr_t[i] != 0:
                k_t[i] = np.log(1+contr_t[i])/contr_t[i]
            else:
                k_t[i] = 1

        carino_smooth = k_t/k * contr

        return np.sum(carino_smooth, axis=1)
        
    def top_bottom_contrib(self, n):
        '''
            returns a dict with the top/bottom n-securities and their log-return contribution 
        '''
        if len(self.tickers) >= n:
            contribution = self.carino_perf_contribution()
            rank = rankdata(contribution, 'ordinal')
            top_idx = [i for i,r in enumerate(rank) if r > len(rank)-n]
            bottom_idx = [i for i,r in enumerate(rank) if r <=n]

            tr = Returns().compute_total_return_idx(self.asset_d_returns) -1
            
            top = {
                self.tickers[i]: {
                    'contribution': contribution[i],
                    'asset_tr': tr[i][-1]
                }
                for i in top_idx
            }
            
            bottom = {
                self.tickers[i]: {
                    'contribution': contribution[i],
                    'asset_tr': tr[i][-1]
                }
                for i in bottom_idx
            }

            return top, bottom
        
        else:
            return None, None