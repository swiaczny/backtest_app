import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
from database import Connect
from scipy.stats import rankdata
from scipy import optimize
# internal
from util import Dates, Misc, Returns, Arrays

class SimParams():

    def __init__(self, start_date, end_date, sector=None, capital=None, weighting=None, reb_freq=None):

        # UNIVERSE
        self.sector = sector

        self.capital = capital
        
        # PF CONSTRUCTION
        self.weighting = weighting
        self.reb_freq = reb_freq

        # DATES        
        self.lookback = self.__set_lookback()
        self.start_date = start_date
        self.end_date = end_date
        self.data_start_date = self.__set_data_start_date()
        self.data_dates = self.__set_data_dates()
        self.sim_dates = self.data_dates[self.lookback:]
        self.trade_dates = self.__set_trade_dates()

    def __set_lookback(self):
        '''
            setting lookback-period (needed for vol calculation)
        '''
        i = 0
        if self.weighting != 'EW':
            i = 63
        
        return i

    def __set_data_start_date(self):
        '''
            setting lookback start_date
        '''
        return Dates().add_business_days(date_ = self.start_date, days = -self.lookback)

    def __set_data_dates(self):
        '''
            returning data-array with weekdays between data_start_date and end_date
        '''
        return np.array(Dates().create_weekday_template(self.data_start_date, self.end_date)['date_'])
    
    def __set_trade_dates(self):
        '''
        out:
            np.array(dt.timestamps)
        '''

        freq_map = {
            'M' : None,
            'Q' : 3,
            'S' : 6,
            'Y' : 12,
            None: None
        }

        df = Dates().last_bday_in_month(self.start_date, self.end_date)
        if self.start_date not in df['date_']:
            add = pd.DataFrame({'date_': dt.strptime(self.start_date, '%Y-%m-%d')}, index=[0])
            df = df.append(add)
            
        trade_dates = [d for d in df['date_']]

        return sorted(trade_dates)[0::freq_map[self.reb_freq]]


class ImportData(SimParams):

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
    
    def __vectorize_prices(self, price_df, tickers):
        '''
            from price_df.col to (mxn) np.array 
        '''
        # (no of assets) x (no of unique dates)
        prices = np.zeros((len(tickers), len(self.data_dates)))
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
            returns     <np.array>  (tickers x date_)
        '''
        tickers = self.__vectorize_tickers(price_df=df)
        prices = self.__vectorize_prices(price_df=df, tickers=tickers)
        returns = self.__vectorize_returns(price_arr=prices)

        return {'tickers': tickers,  'returns': returns}


class PFConstruction(ImportData):

    def __init__(self, start_date, end_date, sector=None, capital=None, weighting=None, reb_freq=None):
        super().__init__(start_date, end_date, sector, capital, weighting, reb_freq)

        data_ = self.vectorize_data(self.import_price_data())

        self.tickers = data_['tickers']
        self.returns_ext = data_['returns']
        self.returns = self.returns_ext[0:,self.lookback:]
    
    def __rolling_vols(self):
        '''
            computing rolling ann.vol

            return mxn np.array with no nan´s (i.e. return-shape equals sim-date-period with no lookback-period)
        '''        
        vol_ext = np.zeros(self.returns_ext.shape)
        vol_sim = np.zeros(self.returns.shape)

        for asset in range(len(self.tickers)):
            vol_ext[asset] = (np.array(pd.Series(self.returns_ext[asset]).rolling(self.lookback).std()))*(252**0.5)
            vol_sim[asset] = vol_ext[asset][self.lookback:]
                
        return vol_sim

    def __equal_weights(self):
        '''
            equal weighting
        '''
        return np.ones(self.returns.shape) / len(self.tickers)

    def __vol_weights(self):
        '''
            inverse vol weighting
        '''
        weights = np.ones(self.returns.shape) / self.__rolling_vols()
        weights = (weights / weights.sum(axis=0))
        
        return weights

    def __optimize(self, trade_date=None):
        '''
            computing target weights for each trade-date
        '''
        from_date = Dates().add_business_days(date_=trade_date, days=-self.lookback)
        returns_lag = Arrays().subset_array(arr=self.returns_ext, dates=self.data_dates, from_date=from_date, to_date=trade_date)
        # SET X0
        weights_eq = np.ones([len(returns_lag)])/len(returns_lag)
        
        # MINIMIZE:
        def min_var(x):
            # to do: shrink it
            cov = np.cov(returns_lag, bias=True)*252
            return np.dot(x, np.dot(cov,x))
        
        # SUBJECT TO:
        def sum_weight(x):
            return np.sum(x)-1

        # BOUNDS
        bound_low = 0.005
        bound_high = 0.10
        b = (bound_low, bound_high)
        bounds = tuple([b]*len(returns_lag))

        # CONSTRAINTS
        constraint = {'type': 'eq', 'fun': sum_weight}

        # optimize
        weights_opt = optimize.minimize(
            fun = min_var, 
            x0 = weights_eq,
            constraints = constraint, 
            bounds = bounds,
            method = 'SLSQP',
            options = {'disp': False}
            )
        
        return weights_opt.x

    def __min_var_weights(self):
        '''
            returning target_weights on trade_dates; rest 0.
        '''
        weights = np.zeros(self.returns.shape)
        for td in self.trade_dates:
            date_idx = Arrays().find_date_index(dates=self.sim_dates, from_date=td, to_date=td)
            weights[0:,date_idx[0][0]] = self.__optimize(trade_date=td)

        return weights

    def target_weights(self, trade_dates=None):
        ''' 
            target weights 
        '''        
        # equal weight
        if self.weighting == 'EW':
            weights = self.__equal_weights()

        # inverse vol-weighting
        if self.weighting == 'VOL':
            weights = self.__vol_weights()
        
        # minimum variance pf
        if self.weighting == 'MINVAR':
            weights = self.__min_var_weights()

        return weights
    

class Simulation(PFConstruction):
    
    def __init__(self, start_date, end_date, sector, capital, weighting, reb_freq):
        super().__init__(start_date, end_date, sector, capital, weighting, reb_freq)

    def simulate(self):
        '''
            iterating through trade-date-periods and simualting performance
        ''' 
        target_weights = self.target_weights()
        
        for i, from_date in enumerate(self.trade_dates):

            if i < len(self.trade_dates)-1:
                to_date = self.trade_dates[i+1]
            else:
                # last trade date, i.e. no end_date
                to_date = None
            
            # SUBSET PRICE DATA ON TIME-PERIOD / COMPUTE TR-INDEX PER ASSET 
            ret = Returns()
            ar = Arrays()
            returns_sub = ar.subset_array(dates=self.sim_dates, arr=self.returns, from_date=from_date, to_date=to_date)
            tr_idx = ret.compute_total_return_idx(returns_sub)            

            # DRIFT MV PER-ASSET WITH NEW WEIGHTS
            if i == 0:
                nav = self.capital
                # select target-allocation on rebalancing date (i.e. from_date)
                target_mv = nav * ar.subset_array(dates=self.sim_dates, arr=target_weights, from_date=from_date, to_date=from_date)
                target_mv = target_mv.squeeze()
                mv_assets = (target_mv * tr_idx.T).T
            else:
                # nav on last available date)                
                nav = sum([asset[-1] for asset in mv_assets])

                # select target-allocation on rebalancing date (i.e. from_date)
                target_mv = nav * ar.subset_array(dates=self.sim_dates, arr=target_weights, from_date=from_date, to_date=from_date)
                target_mv = target_mv.squeeze()
                mv_drift = (target_mv * tr_idx.T).T
                mv_assets = np.append(mv_assets, mv_drift[0:,1:], axis=1)
        
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

    def d_asset_weights(self):
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
        contr = self.d_asset_weights() * self.asset_d_returns
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

