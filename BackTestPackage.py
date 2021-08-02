import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt
import quandl
import requests
import json
import matplotlib.pyplot as plt
from qbstyles import mpl_style

from sklearn import linear_model

quandl.ApiConfig.api_key = "59sg9vqYvngzUw5Xizvi"

API_KEY = 'bd5d7f39831527596de9d85eb60ab188'

Alpha_Vantage_Key = 'XU6NGKGM94P4ICFS'


class UniverseData():
    def __init__(self, sdate, edate):
        self.data_list = []

        self.sdate = sdate
        self.edate = edate
        self.var = sdate
        
        self.data_df = None
        
    def update_yfinance(self, data_list):        
        self.data_list += data_list
        sdate, edate = self.sdate, self.edate
        for ticker in data_list:
            print('retrieving {} from yfinance'.format(ticker))
            while True:
                try:
                    ticker_df = yf.Ticker(ticker).history(
                    	period='max').Close.to_frame().rename(\
                        columns = {'Close':ticker})
                    break
                except:
                    sdate = dt.datetime.strptime(sdate, '%Y-%m-%d')+dt.timedelta(days=1)
                    sdate = sdate.strftime('%Y-%m-%d')
            self.data_df = pd.concat([self.data_df, ticker_df],axis = 1)
            
            
    def update_quandl(self, data_list):
        self.data_list += data_list
        for ticker in data_list:
            print('retrieving {} from Quandl'.format(ticker))
            ticker_df =  quandl.get(ticker, 
            	start_date = self.sdate, end_date = self.edate).rename(\
                columns = {'Value':ticker})
            self.data_df = pd.concat([self.data_df, ticker_df], axis = 1)

    def update_ALFRED(self, data_list):
        self.data_list += data_list

        for ticker in data_list:
            ticker_df = self._get_ALFRED_data(ticker,REAL_TIME_START=self.sdate,
                REAL_TIME_END='9999-12-31')
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

    def _get_ALFRED_data(self, ID, REAL_TIME_START, REAL_TIME_END):
        url = 'https://api.stlouisfed.org/fred/series/observations?series_id={}'.format(ID)
        url += '&realtime_start={}&realtime_end={}&api_key={}&file_type=json'.format(REAL_TIME_START, REAL_TIME_END, API_KEY)
        
        print('retrieving {} from ALFRED'.format(ID))
        print(url)
        
        response = requests.get(url)
        observations = json.loads(response.text)['observations']

        revision_to_date_to_value = {}
        for obs in observations:
            revision = dt.datetime.strptime(obs['realtime_start'], '%Y-%m-%d')
            date =  dt.datetime.strptime(obs['date'], '%Y-%m-%d')
            try :
                value = float(obs['value'])
            except:
                value = np.nan

            try : 
                revision_to_date_to_value[revision][date] = value
            except: 
                revision_to_date_to_value[revision] = {date : value}

        data_df = pd.DataFrame(revision_to_date_to_value).resample('MS').mean().sort_index()
        data_df = data_df.reindex(sorted(data_df.columns), axis = 1)
        data_df = pd.concat({ID:data_df}, axis = 1).T.bfill().T
        data_df = data_df[[data_df.columns[0]]]
        data_df.columns = data_df.columns.get_level_values(0)

        return data_df


class BackTestModule():
    def __init__(self, targets, price_df, sdate, edate, 
        rebal_period = 1, transaction_cost = 0, rebal_bound = 0.05,
        benchmark=None, factors=None):
        self.price_df = price_df
        self.bussiness_date_list = list(self.price_df[['^GSPC']].dropna().index)

        self.target_weights = targets
        self.target_names = list(targets.keys())
        self.main_target = self.target_names[0]
        target_df = targets[self.main_target]

        
        self.universe = list(target_df.columns)
        self.sdate = max(dt.datetime.strptime(sdate, '%Y-%m-%d'), target_df.index[0])
        self.edate = dt.datetime.strptime(edate, '%Y-%m-%d')
        self.rebal_period = rebal_period
        self.transaction_cost = transaction_cost
        self.rebal_bound = rebal_bound

        
        self.asset_df = {target_name : None for target_name in self.target_names}
        self.date_to_target = {target_name : {} for target_name in self.target_names}
        self.weight_df = {target_name : None for target_name in self.target_names}
        self.transaction_df = {target_name : None for target_name in self.target_names}
        self.buy_price_df = {target_name : None for target_name in self.target_names}

        self.benchmark = benchmark
        self.benchmark_name = list(benchmark.keys())[0]
        self.benchmark_ticker = list(benchmark.values())[0]

        self.factors = factors
        self.factor_names = list(factors.keys())
        self.factor_tickers = list(factors.values())

        self.adjust_dates()

        self.setting = {
            'sdate':sdate,
            'edate':edate,
            'rebal_period':rebal_period,
            'transaction_cost':transaction_cost,
        }

        
        self.bm_df = pd.DataFrame({self.benchmark_ticker:[1 for _ in range(len(target_df))]}, index=target_df.index) \
                        if self.benchmark else None
        self.bm_asset_df = None

    def adjust_dates(self):
        while True:
            if self.sdate in self.bussiness_date_list: break
            else: self.sdate += dt.timedelta(days = 1)

        while True:
            if self.edate in self.bussiness_date_list: break
            else: self.edate -= dt.timedelta(days = 1)

    def get_date_to_target(self):
        for target_name in self.target_names:
            for date in self.bussiness_date_list:
                if date < self.sdate:
                    pass
                else:
                    target_date = date
                    while True:
                        try:
                            self.date_to_target[target_name][date] = self.target_weights[target_name].loc[target_date]
                            break
                        except:
                            target_date -= dt.timedelta(days = 1)

    def eval_portfolio(self):
        
        self.get_date_to_target()

        for target_name in self.target_names:
            self.asset_df[target_name], self.transaction_df[target_name], self.buy_price_df[target_name] \
            = self.run_backtest(self.date_to_target[target_name])

        price_df = self.price_df.loc[self.asset_df[self.main_target].index]

        if self.benchmark:
            self.bm_asset_df = price_df[[self.benchmark_ticker]]
            self.bm_asset_df /= self.bm_asset_df.iloc[0]

        self.bm_stat = self.get_stat(asset_df=self.bm_asset_df)
        self.stat = {target_name:self.get_stat(asset_df=self.asset_df[target_name], 
                                        transaction_df=self.transaction_df[target_name], 
                                        BM_asset_df=self.bm_asset_df)
                    for target_name in self.target_names}

        print(self.bm_stat)
        for key in self.stat.keys():
            print(key, self.stat[key])

    def plot_asset(self, ax):
        Ymax = np.max([max(self.asset_df[target_name].sum(axis=1)) for target_name in self.target_names]
            +[max(self.bm_asset_df.sum(axis=1))])
        Ymin = np.min([min(self.asset_df[target_name].sum(axis=1)) for target_name in self.target_names]
            +[min(self.bm_asset_df.sum(axis=1))])
        Ymax += 0.1*(Ymax-1)
        Ymin -= 0.1*(1-Ymin)
        Ymin = min(Ymin, Ymax**(-1/3))

        ax1 = ax
        mdd = ((self.asset_df[self.main_target].sum(axis=1).cummax()-self.asset_df[self.main_target].sum(axis=1))
                / self.asset_df[self.main_target].sum(axis=1).cummax())
        ax1.plot(-mdd, linewidth=0.2,color='gray')
        ax1.fill_between(mdd.index, -mdd.values,color='gray')
        ymin = -1.1*max(mdd)
        ymax = ymin*np.log(Ymax)/np.log(Ymin)
        ax1.set_ylim([ymin,ymax])

        for tick_space in [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]:
            ticks = np.arange(0,ymin,-tick_space)
            if len(ticks) >= 3: break

        ax1.set_yticks(ticks)
        ax1.set_yticklabels(['{}%'.format(round(100*x)) for x in ticks])
        for tick in ticks:
            ax1.axhline(y=tick, color='k', linewidth=0.5, alpha=0.5, zorder=-100)


        ax2=ax1.twinx()
        ax2.plot(self.bm_asset_df.sum(axis=1), linewidth=1.,zorder=100,color='red')
        ax2.plot(self.asset_df[self.main_target].sum(axis=1), linewidth=1.,zorder=200,color='green')
        colors = ['tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan']
        for idx, target_name in enumerate(self.target_names[1:]):
            ax2.plot(self.asset_df[target_name].sum(axis=1), linewidth=0.5, color=colors[idx])


        # outperform = self.asset_df[self.main_target].sum(axis=1)-self.bm_asset_df.sum(axis=1)\
        #             + self.bm_asset_df.sum(axis=1)/self.bm_asset_df.sum(axis=1)
        outperform = self.asset_df[self.main_target].sum(axis=1)/self.bm_asset_df.sum(axis=1)
        plt.fill_between(outperform.index,1,outperform.values,alpha=0.5,zorder=-100,color='green')

        ax2.set_yscale('log')
        ymax = Ymax
        ymin = Ymin
        ax2.set_ylim([ymin, ymax])
        ax2.axhline(y=1, color='k', linewidth=1)
        ax2.legend([self.benchmark_name]+self.target_names, loc='upper left')
        
        for tick_space in [5., 2., 1., 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]:
            ticks = np.arange(1,ymax,tick_space)
            if len(ticks) >= 5: break

        ax2.set_yticks(ticks)
        ax2.set_yticklabels(['{}%'.format(round(100*x-100)) for x in ticks])
        for tick in ticks:
            ax2.axhline(y=tick, color='k', linewidth=0.5, alpha=0.5, zorder=-10)
        

        ax1.margins(x=0)
        ax2.margins(x=0)

        ax1.minorticks_off()
        ax2.minorticks_off()


    def plot_weight(self, ax):
        wgt_df = self.asset_df[self.main_target].copy()
        column_list = list(wgt_df.columns)
        def cum_wgt_df(idx):
            return sum([wgt_df[column_list[len(column_list)-x-1]] for x in range(idx+1)])

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
                'b','g','r','c','m','y',
                'chocolate','goldenrod','olive','honeydew','lime',
                'turquoise','teal','aqua','deepskyblue','dodgerblue']
        try:
            for idx in range(len(column_list)):
                df = cum_wgt_df(len(column_list) - idx - 1)/wgt_df.sum(axis=1)
                ax.plot(df, linewidth = 0.0,color=colors[idx])
                ax.fill_between(df.index, df.values,color=colors[idx])

        except:
            for idx in range(len(column_list)):
                df = cum_wgt_df(len(column_list) - idx - 1)/wgt_df.sum(axis=1)
                ax.plot(df, linewidth = 0.0)
                ax.fill_between(df.index, df.values)


        leg = ax.legend(column_list, ncol=round(len(column_list)/2), 
                        loc='upper center', bbox_to_anchor=(0.5, -0.1),)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        ax.margins(x=0,y=0)

    def plot_performance(self, ax):
        col_labels = list(self.bm_stat.keys())
        row_labels = [self.benchmark_name] + self.target_names
        table_vals = [list(self.bm_stat.values())] \
                    + [list(self.stat[target_name].values()) for target_name in self.target_names]
        table_vals = [ ['{:.1f}%'.format(100*tab[0]), '{:.1f}%'.format(100*tab[1]), 
                    '{:.1f}%'.format(100*tab[2]), '{:.3f}'.format(tab[3]), 
                    '{:.3f}'.format(tab[4]), '{:.0f}%'.format(100*tab[5])]
                    for tab in table_vals ]

        the_table = ax.table(cellText=table_vals,
                              colWidths=[0.05 for _ in col_labels],
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='center',
                              cellColours=[['#122229' for x in tab] for tab in table_vals],
                              rowColours=['#122229' for x in range(len(table_vals))],
                              colColours=['#122229' for x in range(len(table_vals[0]))])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(4, 3)
        ax.axis('off')

    def plot_factor(self, ax):
        price_df = self.price_df.loc[self.asset_df[self.main_target].index]

        if self.factors:
            minlen = min(len(price_df.index)-1,252*3)
            factor_yld_arr = price_df[self.factor_tickers].apply(np.log).diff().iloc[-minlen:].values
            bm_yld_arr = price_df[self.benchmark_ticker].apply(np.log).diff().iloc[-minlen:].values

            factor_exposure = {}
            for target in self.target_names:

                target_yld_arr = self.asset_df[target].sum(axis=1).apply(np.log).diff().iloc[-minlen:].values
                
                try:
                    regr_target = linear_model.LinearRegression()
                    regr_target.fit(factor_yld_arr, target_yld_arr)

                    regr_bm = linear_model.LinearRegression()
                    regr_bm.fit(factor_yld_arr, bm_yld_arr)

                    factor_exposure[target]=regr_target.coef_ - regr_bm.coef_

                except:
                    factor_exposure[target]=np.array([0 for _ in range(len(self.factor_names))])

            factor_df = pd.DataFrame({target:factor_exposure[target] for target in self.target_names},
                index=self.factor_names)

        try:
            colors=['green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan']
            factor_df.plot.barh(ax=ax, color=colors[:len(self.target_names)])
            ax.axvline(x=0,color='k',linewidth=1.)

        except:
            ax.axis('off')
        ax.title.set_text('Factor exposure')

    def plot_report(self, draw_weight=False, draw_factor=False, save_fig=None):
        mpl_style(dark=True)
        price_df = self.price_df.loc[self.asset_df[self.main_target].index]

        if draw_weight and draw_factor:
            scale_factor = 4.
            fig, axs = plt.subplots(2,2,figsize=(3*scale_factor,2*scale_factor),
                                            gridspec_kw={'width_ratios':[2,1]})
            plt.tight_layout(w_pad=2.5*scale_factor)
            
            
            self.plot_weight(axs[1,0])
            self.plot_performance(axs[0,1])
            self.plot_factor(axs[1,1])
            self.plot_asset(axs[0,0])

        else:
            fig, axs = plt.subplots(2,figsize=(10,10))
            self.plot_asset(axs[0])
            self.plot_performance(axs[1])

        if save_fig:
            plt.savefig('{}.png'.format(save_fig))

    def run_backtest(self, date_to_target):
        sdate = self.sdate
        edate = self.edate

        sdate_idx = self.bussiness_date_list.index(sdate)
        return_df = self.price_df/self.price_df.shift()

        date = sdate
        curr_asset = date_to_target[date].copy()
        transaction = curr_asset - curr_asset
        buy_price = curr_asset.copy()

        date_to_asset = {date:curr_asset.copy()}
        date_to_buy_price = {date:curr_asset.copy()}
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
                target_asset = sum(curr_asset) * date_to_target[date]
                assert abs(sum(curr_asset-target_asset)/sum(curr_asset)) < 1e-4

                long_plan = (target_asset-curr_asset).apply(lambda x : np.max([0,x]))
                short_plan = (target_asset-curr_asset).apply(lambda x : np.max([0,-x]))
                transaction_amount = sum(long_plan + short_plan)
                
                if transaction_amount / sum(curr_asset) < self.rebal_bound:
                    target_asset = curr_asset.copy()
                else:
                    long_plan *= (1. - 2 * self.transaction_cost)
                    target_asset = curr_asset + long_plan - short_plan

                    for ticker in buy_price.index:
                        if long_plan[ticker] > 0:
                            buy_price[ticker] += long_plan[ticker]
                        elif short_plan[ticker] > 0:
                            buy_price[ticker] *= (1-(short_plan[ticker]/curr_asset[ticker]))

                    buy_price_long = date_to_buy_price[yesterday] + long_plan
                    buy_price_short = date_to_buy_price[yesterday]*(short_plan/curr_asset)


            else:
                target_asset = curr_asset.copy()

            date_to_transaction[date] = target_asset - curr_asset

            curr_asset = target_asset.copy()

            date_to_asset[date] = curr_asset.copy()

            date_to_buy_price[date] = buy_price.copy()
        
            if date > edate:
                break

        asset_df = pd.DataFrame(date_to_asset).T
        transaction_df = pd.DataFrame(date_to_transaction).T
        buy_price_df = pd.DataFrame(date_to_buy_price).T

        return asset_df, transaction_df, buy_price_df


    def get_stat(self, asset_df, transaction_df=None, BM_asset_df=None):

        return_series = asset_df.sum(axis=1).apply(np.log).diff()[1:]
        CAGR = np.exp(return_series.mean()*252)-1
        VOL = return_series.std()*np.sqrt(252)
        sharpe_ratio = np.sqrt(252)*return_series.mean()/return_series.std()

        if BM_asset_df is not None:
            return_series_BM = BM_asset_df.sum(axis=1).apply(np.log).diff()[1:]
            diff_return_series = return_series - return_series_BM
            information_ratio = np.sqrt(252)*diff_return_series.mean()/diff_return_series.std()
        else:
            information_ratio = 0

        MDD = max((asset_df.sum(axis=1).cummax()-asset_df.sum(axis=1))/
                    asset_df.sum(axis=1).cummax())

        try:
            Turnover = (0.5 * transaction_df.apply(np.abs).sum(axis = 1)/
                    asset_df.sum(axis=1)).resample('y').sum().mean()
        except:
            Turnover = 0

        stat = {
        'CAGR':CAGR,
        'Vol':VOL,
        'MDD':MDD,
        'SR':sharpe_ratio,
        'IR':information_ratio,
        'TO':Turnover
        }

        return stat




                
        
        




