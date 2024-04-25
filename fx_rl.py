import csv
from datetime import datetime, timedelta
from tqdm import tqdm
import random
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_mtsim
sys.path.append("C:/Users/WilliamFetzner/Documents/Trading/")
from gym_mtsim_forked.gym_mtsim.data import FOREX_DATA_PATH, FOREX_DATA_PATH_TRAIN
from gym_mtsim import OrderType, Timeframe, MtEnv, MtSimulator
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import time
import torch
import pickle
import pytz

def get_news_from_csv(News_Trading_Allowed):
    try:
        # Get the date of the past Sunday
        today = datetime.now()
        day_of_week = today.weekday()
        days_to_subtract = 0 if day_of_week == 0 else day_of_week
        past_sunday = today - timedelta(days=days_to_subtract+1)
        
        # Get the month, day, and year of the past Sunday
        month = past_sunday.month
        day = past_sunday.day
        year = past_sunday.year
        
        # Convert the month, day, and year to a string
        month_str = str(month)
        day_str = str(day)
        year_str = str(year)
        
        # Construct the filename
        filename = f"calendar_statement_{year_str}_{month_str}_{day_str}.csv"
        
        any_news = False
        
        with open(filename, 'r', newline='') as file_handle:
            reader = csv.reader(file_handle)
            # Skip the header row
            next(reader)
            for row in reader:
                try:
                    # Assuming "Date" is the first column
                    date = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S') # Adjust the format as per your CSV date format
                    if date.day == today.day:
                        # Find how many hours and minutes until the news event
                        hours = date.hour - today.hour
                        print(hours)
                        if hours == 0:
                            minutes = date.minute - today.minute
                            print(f"News in {minutes} minutes")
                            if not News_Trading_Allowed and abs(minutes) <= 10:
                                print(f"News event happening in {minutes} minutes! No Trades Allowed")
                                any_news = True
                                break
                            elif minutes < -10:
                                print("News event happened within the past hour!")
                                break
                            elif minutes > 10:
                                print("News event happening within an hour")
                                break
                except ValueError:
                    print(f"Error parsing date from row: {row}")
    except FileNotFoundError:
        print(f"Error opening file: {filename}")
    return any_news

def load_env(instrument, path, spread=0.0005, current_balance=200_000., training=False):
    if not training:
        sim = gym_mtsim.MtSimulator(
            unit='USD',
            balance=current_balance,
            leverage=100.,
            stop_out_level=0.2,
            hedge=True,
            symbols_filename=path
        )
        current_time = datetime.now() #+ timedelta(hours=7)
        sim.download_data(
            symbols=['EURUSD', 'AUDCHF', 'NZDCHF', 'GBPNZD', 'USDCAD'],
            time_range=(
                datetime(2024, 1, 1, tzinfo=pytz.UTC),
                current_time
            ),
            timeframe=Timeframe.H1
        )
        sim.save_symbols(path)
    else:
        sim = gym_mtsim.MtSimulator(
            unit='USD',
            balance=current_balance,
            leverage=100.,
            stop_out_level=0.2,
            hedge=True,
            symbols_filename=path
        )
    with open(path, 'rb') as f:
        symbols = pickle.load(f)
    symbols[1][instrument].index = pd.to_datetime(symbols[1][instrument].index)
    max_date = symbols[1][instrument].index.max()

    # what is the day of the week of the max_date
    max_day_of_week = max_date.dayofweek
    # subtract the day of the week from the max_date to get the previous friday
    max_friday = max_date - pd.DateOffset(days=max_day_of_week+2)
    one_week = max_friday - pd.DateOffset(days=7)
    if training:
        training_index_slice = symbols[1][instrument].loc[:one_week, :].index
        fee_ready = lambda symbol: {
            instrument: max(0., np.random.normal(0.0001, 0.00003))
        }[symbol]
    else:
        training_index_slice = symbols[1][instrument].index
        fee_ready = spread

    env = MtEnv(
        original_simulator=sim,
        trading_symbols=[instrument],
        window_size = 10,
        time_points=list(training_index_slice),
        hold_threshold=0.5,
        close_threshold=0.5,
        fee=fee_ready,
        symbol_max_orders=2,
        multiprocessing_processes=2
    )

    return env