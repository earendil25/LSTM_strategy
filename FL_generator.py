import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import os

from sklearn.neighbors import KernelDensity

from BackTestPackage import UniverseData

class FL_generator():
    def __init__(self, UniverseData):
        self.dataset = UniverseData

    def make_feature_label_df(self, config):

        self.feature_config = config

        benchmark = config['benchmark']
        universe_list = config['universe_list']
        input_info = config['inputs']
        forward_day = config['forward_day']

        inputs = input_info.keys()

        date_to_label = {}
        input_to_date_to_feature = { name:{} for name in inputs }

        input_to_df = {}

        for ipt in inputs:
            feature_list = input_info[ipt]['feature_list']
            period = input_info[ipt]['period']
            input_to_df[ipt] = self.dataset.data_df[feature_list].resample(period
                                            ).mean()

        date_list = list(input_to_df['daily_price'].dropna().index)

        date_to_ipt_to_feature = {}
        for date in date_list:
            date_idx = date_list.index(date)

            ## Update label ####################################################
            try:
                future_date = date_list[date_idx + forward_day]

                yld_list = [(input_to_df['daily_price'].loc[future_date][ticker] \
                            / input_to_df['daily_price'].loc[date][ticker])
                            for ticker in universe_list]

                yld_cut_BM = (input_to_df['daily_price'].loc[future_date][benchmark] \
                            / input_to_df['daily_price'].loc[date][benchmark])

                top_num = (int)(len(universe_list)*0.5)
                yld_cut_top = sorted(yld_list, reverse = True)[top_num]

                yld_cut = np.max([yld_cut_BM, yld_cut_top])

                label_list = [1.0*(yld > yld_cut) for yld in yld_list]

                assert np.isnan(sum(label_list)) == False

                label = label_list # / sum(label_list)

            except:
                label = np.nan

            ## Update feature ############################################
            ipt_to_feature = {}
            for ipt in inputs:
                ipt_df = input_to_df[ipt]

                feature_date_list = list(ipt_df.dropna().index)

                feature_list = input_info[ipt]['feature_list']
                period = input_info[ipt]['period']
                lookback_window = input_info[ipt]['lookback_window']

                try:
                    edate = date
                    itr_checker = 0
                    while itr_checker < 100:
                        itr_checker += 1
                        if edate in feature_date_list:
                            edate_idx = feature_date_list.index(edate)
                            if period == 'm': edate_idx -= 1
                            break
                        else:
                            edate -= dt.timedelta(days = 1)

                    assert itr_checker < 100

                    assert edate_idx - lookback_window[0] >= 0

                    history_window = feature_date_list[edate_idx - lookback_window[0]:edate_idx]
                    feature_array = ipt_df.loc[history_window][feature_list].values


                    if np.isnan(np.sum(feature_array)):
                        feature_array = np.nan

                    feature_array = np.diff(
                        tf.keras.layers.LayerNormalization(axis=0)(feature_array)[-lookback_window[1]-1:],
                        axis = 0)

                except:
                    feature_array = np.nan


                ipt_to_feature[ipt] = feature_array

            date_to_label[date] = label
            date_to_ipt_to_feature[date] = ipt_to_feature

        label_list = [date_to_label[date] for date in date_list]

        ipt_to_feature_list = {}
        for ipt in inputs:
            ipt_to_feature_list[ipt] = [date_to_ipt_to_feature[date][ipt] for date in date_list]

        feature_label_dic = {'label':label_list}
        feature_label_dic.update({ipt:ipt_to_feature_list[ipt] for ipt in inputs})

        feature_label_df = pd.DataFrame(feature_label_dic, index=date_list)

        for ipt in inputs:
            feature_label_df = feature_label_df[feature_label_df[ipt].notna()]

        self.feature_label_df = feature_label_df

        return feature_label_df



    