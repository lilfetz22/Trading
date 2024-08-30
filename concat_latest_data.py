import numpy as np
import pandas as pd
import polars as pl
import pickle
from datetime import datetime, timedelta
import sys
from mlforecast import MLForecast
sys.path.append("C:/Users/WilliamFetzner/Documents/Trading/")
from gym_mtsim_forked.gym_mtsim.data import FOREX_DATA_PATH
import fx_rl

def check_for_new_news():
    news_data_full = (
        pl.scan_csv('calendar_df_full_updated.csv')
        .with_columns([
            pl.col('datetime').str.to_datetime(),
        ])
        .select('Id', 'datetime')
    ).collect()
    max_news_date = news_data_full.select(pl.col('datetime').max()).item()
    try:
        news_data = (
            pl.scan_csv('calendar-event-list.csv')
            .with_columns([
                pl.col('Start').alias('datetime').str.to_datetime(format='%m/%d/%Y %H:%M:%S').dt.offset_by('7h'),
            ])
            .select('Id', 'datetime')
        ).collect()
        max_new_news_date = news_data.select(pl.col('datetime').max()).item()
        if max_news_date < max_new_news_date:
            # concatenate the news data with news_data_full
            print('adding latest data')
            news_data_full = pl.concat([news_data_full, news_data])
            news_data_full.write_csv('calendar_df_full_updated.csv')
    except:
        print('no news data found')
    
    return news_data_full

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
    # reset the index
    latest_data.reset_index(inplace=True)
    latest_data_cols_added = add_exog_columns(latest_data, symbols_1hr[1]['EURUSD'], 'lgb')
    # drop duplicates in the index
    latest_data_cols_added.drop_duplicates(keep='last', inplace=True)
    # set the index name
    latest_data_cols_added.index.name = 'Time'
    latest_data_cols_added.index = pd.to_datetime(latest_data_cols_added.index, utc=True)
    symbols_1hr[1]['EURUSD'] = latest_data_cols_added

    with open(FOREX_DATA_PATH, 'wb') as f:
        pickle.dump(symbols_1hr, f)

    return 'Latest data added to symbols_1hr'


def find_seconds_to_next_news(df, news_counts):
    # Ensure both DataFrames have their time columns as datetime
    df = df.with_columns(pl.col("ds").cast(pl.Datetime))
    news_counts = news_counts.with_columns(pl.col("datetime").cast(pl.Datetime))

    # Combine both DataFrames
    combined = pl.concat([
        df.select(pl.col("ds").alias("time")).with_columns(pl.lit("main").alias("source")),
        news_counts.select(pl.col("datetime").alias("time")).with_columns(pl.lit("news").alias("source"))
    ])

    # Sort by time, source, time ascending, source descending
    combined = combined.sort(["time", "source"], descending=[False, True])

    # Find the next news event for each row
    result = combined.with_columns([
        pl.when(pl.col("source") == "news")
        .then(pl.col("time"))
        .otherwise(None)
        .forward_fill()
        .alias("prev_news_time"),
        pl.when(pl.col("source") == "news")
        .then(pl.col("time"))
        .otherwise(None)
        .backward_fill()
        .alias("next_news_time"),
        pl.col("source")
    ])

    # Calculate the time difference for main events
    result = result.filter(pl.col("source") == "main").with_columns([
        pl.col('time').dt.week().alias('time_week_nbr'),
        pl.col('prev_news_time').dt.week().alias('prev_time_week_nbr'),
        pl.col('next_news_time').dt.week().alias('next_time_week_nbr'),
        (
            pl.when(
                (pl.col("time").dt.week() != pl.col("prev_news_time").dt.week())
            )
            .then((pl.col("prev_news_time") - pl.col("time")).dt.total_seconds() + (2 * 86400))
            .otherwise((pl.col("prev_news_time") - pl.col("time")).dt.total_seconds())
        ).alias("seconds_since_last_news_event"),
        # get the week number for "time"
        

        (
            pl.when(
                (pl.col("time").dt.week() != pl.col("next_news_time").dt.week())
            )
            .then((pl.col("next_news_time") - pl.col("time")).dt.total_seconds() - (2 * 86400))
            .otherwise((pl.col("next_news_time") - pl.col("time")).dt.total_seconds())
        ).alias("seconds_to_next_news_event")
    ])


    # Join the result back to the original DataFrame
    final_df = df.join(
        result.select(pl.col("time").alias("ds"), "seconds_since_last_news_event", "seconds_to_next_news_event"), 
        on="ds", 
        how="left"
    )

    return final_df


def add_exog_columns(df, full_df, best_model_name):
    # convert to polars dataframe
    pl_df = pl.from_pandas(df).with_columns([
        # add a unique_id col which will be 'EURUSD' as a string
        pl.lit('EURUSD').alias('unique_id').cast(pl.Utf8),
        pl.col('Time').alias('ds').str.to_datetime(format='%Y-%m-%d %H:%M:%S+00:00'),
        # .dt.replace_time_zone('UTC'),
        # add a column that takes the sum of 'Open' 'High' 'Low' and 'Close' and divides it by 4
        ((pl.col('Open').cast(pl.Float64) + pl.col('High').cast(pl.Float64) + pl.col('Low').cast(pl.Float64) + pl.col('Close').cast(pl.Float64)) / 4).round(5).alias('y')
    ]).sort('ds').select(pl.col('unique_id', 'ds', 'y'))

    news_data = check_for_new_news()
    news_counts = news_data.group_by('datetime').len().sort(by='datetime')

    df_w_news = find_seconds_to_next_news(pl_df, news_counts)
    df_w_news_new_ds = df_w_news.with_columns([
        pl.col('ds').alias('Time'),
        pl.col('y').alias('Close')
    ]).drop('unique_id')

    # adjust full_df
    full_df_w_new_data = pl.concat([
        full_df.select(*df_w_news_new_ds.columns), 
        df_w_news_new_ds])
    full_df_w_new_data_ds = full_df_w_new_data.with_columns([
            pl.arange(0, full_df.height).alias('ds')
        ])
    full_df_adj = full_df_w_new_data_ds.with_columns([
            pl.col('Close').alias('y'),
            pl.lit('EURUSD').alias('unique_id').cast(pl.Utf8),
        ]).drop('Time')

        

    best_model = MLForecast.load(f'best_models2/{best_model_name}') # change 'lgb' to the actual best model name
    best_model_cv_df = best_model.cross_validation(
        df=full_df_adj,
        h=1,
        n_windows=len(df_w_news_new_ds),
        static_features=[]
    )
    best_model_cv_df_sel = best_model_cv_df.select(pl.col('cutoff').alias('ds'), pl.col(best_model_name))
    # merge with df_w_news_new_ds
    model_preds_added = full_df_w_new_data_ds.join(best_model_cv_df_sel, on='ds', how='inner')
    all_data = pl.concat([full_df, model_preds_added])

    return all_data.set_index('Time').sort_index()



# columns from full_df
# Time index, Close, best_model_name, seconds_since_last_news_event, seconds_to_next_news_event