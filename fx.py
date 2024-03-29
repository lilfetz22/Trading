import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import datetime as dt
import mplfinance 
from renkodf import Renko
from scipy.signal import lfilter

def test_fx(x):
    x = x + 1
    return x

def calc_smma(src, smma_len):
    smma = np.empty_like(src)
    smma[0] = src[0]
    for i in range(1, len(src)):
        smma[i] = smma[i-1] if np.isnan(smma[i-1]) else (smma[i-1] * (smma_len - 1) + src[i]) / smma_len
    return smma


def calc_zlema(src, length):
    ema1 = pd.Series(src).ewm(span=length).mean().values
    ema2 = pd.Series(ema1).ewm(span=length).mean().values
    d = ema1 - ema2
    return ema1 + d

def calc_impulse_macd(df, lengthMA=34, lengthSignal=9):
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['hi'] = calc_smma(df['high'].values, lengthMA)
    df['lo'] = calc_smma(df['low'].values, lengthMA)
    df['mi'] = calc_zlema(df['hlc3'].values, lengthMA)
    df['md'] = np.where(df['mi'] > df['hi'], df['mi'] - df['hi'], np.where(df['mi'] < df['lo'], df['mi'] - df['lo'], 0))
    df['sb'] = df['md'].rolling(window=lengthSignal).mean()
    df['sh'] = df['md'] - df['sb']
    # find the sign of sh
    df['sh_sign'] = np.sign(df['sh'])
    # indicate the trade signal for Impulse
    df['impulse_signal'] = np.where(df['sh_sign'] == 1, 'buy', np.where(df['sh_sign'] == -1, 'sell', 'none'))
    return df

class PSAR:
   def __init__(self, start=0.02, inc=0.02, max=0.2):
       self.max_af = max
       self.init_af = start
       self.af = start
       self.af_step = inc
       self.extreme_point = None
       self.high_price_trend = []
       self.low_price_trend = []
       self.high_price_window = deque(maxlen=2)
       self.low_price_window = deque(maxlen=2)
       self.psar_list = []
       self.af_list = []
       self.ep_list = []
       self.high_list = []
       self.low_list = []
       self.trend_list = []
       self._num_days = 0

   def calcPSAR(self, high, low):
       if self._num_days >= 3:
           psar = self._calcPSAR()
       else:
           psar = self._initPSARVals(high, low)

       psar = self._updateCurrentVals(psar, high, low)
       self._num_days += 1

       return psar

   def _initPSARVals(self, high, low):
       if len(self.low_price_window) <= 1:
           self.trend = None
           self.extreme_point = high
           return None

       if self.high_price_window[0] < self.high_price_window[1]:
           self.trend = 1
           psar = min(self.low_price_window)
           self.extreme_point = max(self.high_price_window)
       else: 
           self.trend = 0
           psar = max(self.high_price_window)
           self.extreme_point = min(self.low_price_window)

       return psar

   def _calcPSAR(self):
       prev_psar = self.psar_list[-1]
       if self.trend == 1: # Up
           psar = prev_psar + self.af * (self.extreme_point - prev_psar)
           psar = min(psar, min(self.low_price_window))
       else:
           psar = prev_psar - self.af * (prev_psar - self.extreme_point)
           psar = max(psar, max(self.high_price_window))

       return psar

   def _updateCurrentVals(self, psar, high, low):
       if self.trend == 1:
           self.high_price_trend.append(high)
       elif self.trend == 0:
           self.low_price_trend.append(low)

       psar = self._trendReversal(psar, high, low)

       self.psar_list.append(psar)
       self.af_list.append(self.af)
       self.ep_list.append(self.extreme_point)
       self.high_list.append(high)
       self.low_list.append(low)
       self.high_price_window.append(high)
       self.low_price_window.append(low)
       self.trend_list.append(self.trend)

       return psar

   def _trendReversal(self, psar, high, low):
       reversal = False
       if self.trend == 1 and psar > low:
           self.trend = 0
           psar = max(self.high_price_trend)
           self.extreme_point = low
           reversal = True
       elif self.trend == 0 and psar < high:
           self.trend = 1
           psar = min(self.low_price_trend)
           self.extreme_point = high
           reversal = True

       if reversal:
           self.af = self.init_af
           self.high_price_trend.clear()
           self.low_price_trend.clear()
       else:
           if high > self.extreme_point and self.trend == 1:
               self.af = min(self.af + self.af_step, self.max_af)
               self.extreme_point = high
           elif low < self.extreme_point and self.trend == 0:
               self.af = min(self.af + self.af_step, self.max_af)
               self.extreme_point = low

       return psar

def psar_from_data(df, increment, maximum):
    # Calculate PSAR
    high = np.array(df['high']) # replace with actual high prices
    low = np.array(df['low']) # replace with actual low prices
    close = np.array(df['close']) # replace with actual closing prices
    # I don't have start in the indicator I'm using in mt4 so increment will be used for start as well
    psar_obj = PSAR(increment, increment, maximum) 
    psar = np.empty_like(high)
    for i in range(len(high)):
        psar[i] = psar_obj.calcPSAR(high[i], low[i])
    # Determine direction
    dir = np.where(psar < close, 'buy', 'sell')
    df['psar'] = psar
    df['psar_signal'] = dir

    return df

def add_swap_rates(df, qcr, bcr, lots, acct, remove_neg_swap=True):
    # find the positions where the period in between the entry_time and exit_time include 5:00 pm 
    df[acct + '_entry_time'] = pd.to_datetime(df[acct + '_entry_time'])
    df[acct + '_exit_time'] = pd.to_datetime(df[acct + '_exit_time'])
    no_nas = df.dropna(subset=[acct + '_entry_time', acct + '_exit_time'])
    no_nas['entry_time_t'] = no_nas[acct + '_entry_time'].dt.time
    no_nas['exit_time_t'] = no_nas[acct + '_exit_time'].dt.time
    no_nas['entry_time_str'] = no_nas['entry_time_t'].astype(str)
    no_nas['exit_time_str'] = no_nas['exit_time_t'].astype(str)
    no_nas['entry_time_hr'] = no_nas['entry_time_str'].str.split(':').str[0].astype(int)
    no_nas['exit_time_hr'] = no_nas['exit_time_str'].str.split(':').str[0].astype(int)
    # if 23 is between entry_time_hr and exit_time_hr then add a column called 'swap' and set it to 1
    no_nas[acct + '_swap'] = np.where((no_nas['entry_time_hr'] < 23) & (no_nas['exit_time_hr'] >= 23), 1, 0)
    # drop entry_time_t, exit_time_t, entry_time_str, exit_time_str, entry_time_hr, exit_time_hr
    no_nas.drop(columns=['entry_time_t', 'exit_time_t', 'entry_time_str', 'exit_time_str', 'entry_time_hr', 'exit_time_hr'], inplace=True)
    no_nas[acct + '_swap_rate'] = np.where((no_nas[acct + '_direction'].str.strip() == 'buy') & (no_nas[acct + '_swap'] == 1), 
                                      (lots*100000*(qcr-bcr))/(365 * no_nas[acct + '_exit_price']),
                                    np.where((no_nas[acct + '_direction'].str.strip() == 'sell')  & (no_nas[acct + '_swap'] == 1), 
                                             (lots*100000*(bcr-qcr))/(365 * no_nas[acct + '_exit_price']), 0))
    if remove_neg_swap:
        no_nas[acct + '_swap_rate'] = np.where(no_nas[acct + '_swap_rate'] < 0, 0, no_nas[acct + '_swap_rate'])
    
    # only grab the swap and swap_rate columns for no_nas
    no_nas_fil = no_nas.loc[:, [acct + '_swap', acct + '_swap_rate']]
    
    # join no_nas back with df
    df = df.join(no_nas_fil)
    
    return df

## Data prep
def prep_data(df, year):
    # find the year of df.datetime
    df['year'] = df['datetime'].dt.year
    # filter the df to just the year
    df_year = df[df['year'] == year]
    
    # drop the volume column
    df_year = df_year.drop(columns=['volume'])
    df_year = df_year.set_index('datetime')
    return df_year

def cum_count(df):
    # position_count will be a cumulative count used to filter the data to the timeframe between the entry 
    # and exit signals so anytime there is an "entry + buy" or "entry + short" the count should increase by 1
    df['position_count'] = np.where(df['sma_crossover'] == 1, 1, np.where(df['sma_crossover'] == -1, 1,0))
    df['nova_cum_position_count'] = df['position_count'].cumsum()
    # if day_of_week_transition is 1 then make the cum_position_count null
    df['nova_cum_position_count'] = np.where(df['sma_signal'] == 'exit', None, df['nova_cum_position_count'])
    # a new column called msolutions_cum_position_count which is the cum_position_count but only for msolutions
    df['msolutions_cum_position_count'] = np.where(df['news_event_5'] != 1, df['nova_cum_position_count'], None)
    return df

def add_tp_sl(df, take_profit, stop_loss):
    df['take_profit'] = np.where(df['sma_crossover'] == 1, df['open'] + take_profit, 
                                 np.where(df['sma_crossover'] == -1, df['open'] - take_profit, np.nan))
    df['stop_loss'] = np.where(df['sma_crossover'] == 1, df['open'] - stop_loss, 
                                 np.where(df['sma_crossover'] == -1, df['open'] + stop_loss, np.nan))
    return df