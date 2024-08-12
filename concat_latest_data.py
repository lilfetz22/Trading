import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import sys
sys.path.append("C:/Users/WilliamFetzner/Documents/Trading/")
from gym_mtsim_forked.gym_mtsim.data import FOREX_DATA_PATH
import fx_rl

with open(FOREX_DATA_PATH, 'rb') as f:
    symbols_1hr = pickle.load(f)

def prep_data_fx(df):
    df = df.reset_index(drop=True)
    # drop any duplicate rows
    df = df.drop_duplicates()
    # convert date to datetime
    df['datetime'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d %H:%M:%S:%f')
    # sort by datetime
    df = df.sort_values(by='datetime')
    # rename bid price to close
    df.rename(columns={'Bid price':'close'}, inplace=True)
    df_ready = df.set_index('datetime')
    # adjust the datetime 7 hrs ahead to match market time
    df_ready.index = df_ready.index + pd.Timedelta(hours=7)
    return df_ready

def get_latest_data_fx():
    try:
        start = (datetime.now() - timedelta(days=7)).strftime('%m_%d')
        end = datetime.now().strftime('%m_%d_%y')
        df = pd.read_csv(f'EURUSD_{start}_to_{end}.csv')
    except:
        return f'file for {start} to {end} does not exist.'
    
    latest_data = prep_data_fx(df)
    # latest_data.index = pd.to_datetime(latest_data.index)
    # latest_data = latest_data.set_index('datetime')
    latest_data.drop(columns={'Timestamp', 'Ask price', 'Bid volume', 'Ask volume'}, inplace=True)
    latest_data = latest_data.resample('H').ohlc()
    # # drop the first index of the column multiindex
    latest_data.columns = latest_data.columns.droplevel(0)
    # the index needs to be called "Time"
    latest_data.index.name = "Time"
    # capitalize the column names
    latest_data.columns = latest_data.columns.str.capitalize()

    symbols_1hr[1]['EURUSD'] = pd.concat([symbols_1hr[1]['EURUSD'], latest_data]).sort_index()
    # drop duplicates in the index
    symbols_1hr[1]['EURUSD'].drop_duplicates(keep='last', inplace=True)
    # set the index name
    symbols_1hr[1]['EURUSD'].index.name = 'Time'
    symbols_1hr[1]['EURUSD'].index = pd.to_datetime(symbols_1hr[1]['EURUSD'].index, utc=True)
    with open(FOREX_DATA_PATH, 'wb') as f:
        pickle.dump(symbols_1hr, f)
    return 'Latest data added to symbols_1hr'