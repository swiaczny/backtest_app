from datetime import datetime as dt, timedelta
import pandas as pd
import numpy as np

class Dates():
    
    def create_weekday_template(self, start_date, end_date):
        '''
            creates a dataframe with weekdays between and including start- and end-date

            input:
                start_date, end_date <string>
            output:
                DataFrame:
                columns
                    date_ <dt.datetime>  
        '''
        df = pd.DataFrame()
        df['date_'] = pd.date_range(start_date, end_date, freq='D').to_series()
        df['weekday'] = df['date_'].dt.dayofweek
        weekday = df.loc[df['weekday'] < 5, 'date_'].copy()
        odf = pd.DataFrame(weekday)
        odf.reset_index(drop=True, inplace=True)
    
        return odf

    def add_business_days(self, date_, days):
        '''
            adds business day as datetime object.
            
            input:
                date_ <str or dt.datetime>
                days <int>
            output
                <dt.datetime>
        '''
            
        if not isinstance(date_, dt):
            try:
                date_ = dt.strptime(date_, '%Y-%m-%d')
            except:
                print("failed to parse {date_} as date")
                return None

        if abs(days) >0:

            counter = abs(days)

            d = 0
            while counter > 0:
                if days >0:
                    d += 1
                else:
                    d -= 1
                    
                next_date = date_ + timedelta(days=d)
                if next_date.weekday()<5:
                    counter -= 1

            return next_date
        
        else:
            return date_
     
    def last_bday_in_month(self, start_date, end_date):
        '''
            returns a df with month-end business dates between and incl. start_ and end_date

            input:
                start_date, end_date <string>
            output:
                DataFrame:
                columns
                    date_ <dt.datetime>  
        '''
        # extening end_date to avoid having end_date as month_end automatically 
        end_date_ext = self.add_business_days(date_=end_date, days=1)
        
        # weekday template
        df = self.create_weekday_template(start_date, end_date_ext)
        df['day'] = df['date_'].dt.day
        df['month'] = df['date_'].dt.month
        df['year'] = df['date_'].dt.year

        # find max(day) given year and month 
        df.set_index(df['date_'], inplace=True)
        df = df[['date_', 'day', 'month', 'year']].groupby(['month','year']).transform('max')
        df = df.loc[df['date_']==df.index].copy()
    
        # remove end_date_ext
        df = df.loc[df['date_']!=end_date_ext]
        df.reset_index(drop=True, inplace=True)
    
        return df[['date_']]

    def str_to_date(self, df, col):
        """
            input = (df, colname)
            output = df

            formats date-column from string to datetime (midnight)
        """
        df[col] = df[col].apply(lambda x : pd.Timestamp(x).normalize())

        return df
    

class Returns():
 
    def compute_d_returns(self, prices):
        '''
            price to daily return
            out: (no-of-securities x dates) np.array
        '''
        # prices = more-dim-array
        if len(prices.shape) > 1:
            returns = np.zeros(prices.shape)

            for i in range(prices.shape[0]):
                returns[i][1:] = np.diff(prices[i]) / prices[i][:-1]

        # prices = one-dim-array
        else:
            returns = np.zeros(prices.shape[0])
            for i in range(prices.shape[0]-1):
                returns[i+1] = prices[i+1] / prices[i] -1

        return returns

        def compute_total_return_idx(self, returns):
            '''
                compound returns
            '''
            # more-dim-array
            if len(returns.shape) >1:
                tr = np.ones(returns.shape)

                for i in range(returns.shape[0]):
                    tr[i][0:] = np.cumprod(1+returns[i])

            # one-dim-array
            else:
                tr = np.cumprod(1+returns)
                
            return tr

    def compute_total_return_idx(self, returns):
        '''
            compound returns
        '''
        # more-dim-array
        if len(returns.shape) >1:
            tr = np.ones(returns.shape)

            for i in range(returns.shape[0]):
                tr[i][0:] = np.cumprod(1+returns[i])

        # one-dim-array
        else:
            tr = np.cumprod(1+returns)
            
        return tr
    
    def impute_with_mean(self, return_arr):
        '''
            imputing missing values with mean
        '''
        ret_mean = np.nanmean(return_arr, axis=0)
        idx_nan = np.where(np.isnan(return_arr))
        return_arr[idx_nan] = np.take(ret_mean, idx_nan[1])

        return return_arr


class Arrays():

    def find_date_index(self, dates, from_date, to_date=None):
        '''
        '''
        if to_date is not None:
            date_idx = np.where((dates >= from_date) & (dates <= to_date))
        else:
            date_idx = np.where(dates >= from_date)

        return date_idx

    def subset_array(self, arr, dates, from_date, to_date=None):
        '''
            subsets array on dates
        '''
        date_idx = self.find_date_index(dates=dates, from_date=from_date, to_date=to_date)

        # more-dim-array
        if len(arr.shape) > 1:
            # arr_sub = np.zeros((arr.shape[0], len(date_idx)))
            arr_sub = np.zeros(arr.shape)

            for i in range(arr.shape[0]):
                if i == 0:
                    arr_sub = np.array([arr[i][date_idx]])
                else:
                    arr_sub = np.r_[arr_sub, [arr[i][date_idx].T]]

            return arr_sub


class Misc():
    '''
        misc helper methods
    '''
    def list_to_sql(self, input_list):
        '''
            transforms list or set to sql-string
            input = ['a','b','c']
            output_string = 'a','b','c'
        '''
        str_out = ""
        for i in input_list:
            str_out += "'"+str(i)+"'"+','

        return str_out[:-1]
