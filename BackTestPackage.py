import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import quandl

quandl.ApiConfig.api_key = "59sg9vqYvngzUw5Xizvi"

class UniverseData():
    def __init__(self, sdate, edate):
        self.data_list = []

        self.sdate = sdate
        self.edate = edate
        self.var = sdate
        
        self.data_df = None
        
    def update_yfinance(self, data_list):        
        self.data_list += data_list
        for ticker in data_list:
            ticker_df = yf.Ticker(ticker).history(
            	start = self.sdate, end = self.edate).Close.to_frame().rename(\
                columns = {'Close':ticker})
            self.data_df = pd.concat([self.data_df, ticker_df],axis = 1)
            
    def update_quandl(self, data_list):
        self.data_list += data_list

        for ticker in data_list:
            ticker_df =  quandl.get(ticker, 
            	start_date = self.sdate, end_date = self.edate).rename(\
                columns = {'Value':ticker})
            self.data_df = pd.concat([self.data_df, ticker_df], axis = 1)

    def combine(self, col1, col2):
        df = self.data_df
        idx = df[[col1]].dropna().index[0]
        adj_factor = df[[col1]].loc[idx].values[0]/df[[col2]].loc[idx].values[0]
        df[[col2]] = df[[col2]]*adj_factor
        
        data_list = []
        for date in df.index:
            val1 = df[col1].loc[date]
            val2 = df[col2].loc[date]
            if np.isnan(val1) == False:
                data_list.append(val1)
            else : data_list.append(val2)
                
        df = df.drop([col1], axis =1)
        df.insert(0, col1, data_list)

        self.data_df = df



class BackTestModule():
    def __init__(self, target_df, price_df, sdate, edate, 
        rebal_period = 1, transaction_cost = 0, rebal_bound = 0.05):
        self.price_df = price_df
        self.bussiness_date_list = list(self.price_df[['^GSPC']].dropna().index)
        self.target_df = target_df
        self.universe = list(target_df.columns)
        self.sdate = max(dt.datetime.strptime(sdate, '%Y-%m-%d'), target_df.index[0])
        self.edate = dt.datetime.strptime(edate, '%Y-%m-%d')
        self.rebal_period = rebal_period
        self.transaction_cost = transaction_cost
        self.rebal_bound = rebal_bound

        self.adjust_dates()

        self.setting = {
            'sdate':sdate,
            'edate':edate,
            'rebal_period':rebal_period,
            'transaction_cost':transaction_cost,
        }

        self.asset_df = None
        self.weight_df = None
        self.transaction_df = None

        self.date_to_target = {}
        for date in self.bussiness_date_list:
            if date < self.sdate:
                pass
            else:
                target_date = date
                while True:
                    try:
                        self.date_to_target[date] = target_df.loc[target_date]
                        break
                    except:
                        target_date -= dt.timedelta(days = 1)

    def adjust_dates(self):
        while True:
            if self.sdate in self.bussiness_date_list:
                break
            else:
                self.sdate += dt.timedelta(days = 1)

        while True:
            if self.edate in self.bussiness_date_list:
                break
            else:
                self.edate -= dt.timedelta(days = 1)

        print(self.sdate, self.edate)


    def run_backtest(self):
        sdate = self.sdate
        edate = self.edate

        sdate_idx = self.bussiness_date_list.index(sdate)
        return_df = self.price_df/self.price_df.shift()

        date = sdate
        curr_asset = self.date_to_target[date].copy()
        transaction = curr_asset - curr_asset

        date_to_asset = {date:curr_asset.copy()}
        date_to_transaction = {date:transaction}

        t_step = 0
        for date in self.bussiness_date_list[sdate_idx + 1:]:
            t_step += 1
            is_rebal = False
            date_idx = self.bussiness_date_list.index(date)
            yesterday = self.bussiness_date_list[date_idx-1]
            ret = self.price_df.loc[date]/self.price_df.loc[yesterday]
            curr_asset *= ret

            if self.rebal_period == 'M':
                date_idx = self.bussiness_date_list.index(date)
                yesterday = self.bussiness_date_list[date_idx - 1]
                is_rebal = (date.month != yesterday.month)
            else:
                is_rebal = (t_step % self.rebal_period == 0)

            if is_rebal:
                target_asset = sum(curr_asset) * self.date_to_target[date]
                assert abs(sum(curr_asset-target_asset)/sum(curr_asset)) < 1e-4
                transaction_amount = sum((target_asset-curr_asset).apply(np.abs))
                
                if transaction_amount / sum(curr_asset) < self.rebal_bound:
                    target_asset = curr_asset.copy()
                else:
                    transaction_fee = transaction_amount*self.transaction_cost
                    target_asset = (sum(curr_asset)-transaction_fee) * self.date_to_target[date]
            else:
                target_asset = curr_asset.copy()

            date_to_transaction[date] = target_asset - curr_asset

            curr_asset = target_asset.copy()

            date_to_asset[date] = curr_asset.copy()
        
            if date > edate:
                break

        asset_df = pd.DataFrame(date_to_asset).T
        transaction_df = pd.DataFrame(date_to_transaction).T

        self.asset_df = asset_df
        self.transaction_df = transaction_df

    def get_stat(self, BM_asset_df = None):

        num_days = (self.asset_df.sum(axis=1).index[-1]-self.asset_df.sum(axis=1).index[0]).days
        CAGR = np.exp(np.log(self.asset_df.sum(axis=1)[-1])*252/num_days)-1

        return_series = self.asset_df.sum(axis=1).resample('M').mean().apply(np.log).diff()[1:]
        sharpe_ratio = np.sqrt(12)*return_series.mean()/return_series.std()

        if BM_asset_df is not None:
            return_series_BM = BM_asset_df.sum(axis=1).resample('M').mean().apply(np.log).diff()[1:]
            diff_return_series = return_series - return_series_BM
            information_ratio = np.sqrt(12)*diff_return_series.mean()/diff_return_series.std()
        else:
            information_ratio = None

        MDD = max((self.asset_df.sum(axis=1).cummax()-self.asset_df.sum(axis=1))/self.asset_df.sum(axis=1).cummax())

        stat = {
        'CAGR':CAGR,
        'sharpe_ratio':sharpe_ratio,
        'information_ratio':information_ratio,
        'MDD':MDD
        }

        return stat




                
        
        




