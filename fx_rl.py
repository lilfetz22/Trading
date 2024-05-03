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

def get_news_from_csv(News_Trading_Allowed, STime):
    try:
        # Get the date of the past Sunday
        today = STime
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
        location = "C:/Users/WilliamFetzner/AppData/Roaming/MetaQuotes/Terminal/0C2BFA140CA8FBEFEDCADDDEDD61AA24/tester/files/calendar_statement_"
        filename = f"{location}{year_str}_{month_str}_{day_str}.csv"
        
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
                        # print(hours)
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

## add the latest data to the environment
def get_latest_data(path_to_data, new_data, instrument='EURUSD'):
    with open(path_to_data, 'rb') as f:
        symbols = pickle.load(f)
    current_data = symbols[1][instrument]
    # remove the last row of current data
    current_data = current_data.iloc[:-1, :]
    # Capitalize the column names in new_data
    new_data.columns = [col.capitalize() for col in new_data.columns]
    # set the date column to be "Time" and set it as the index
    new_data = new_data.rename(columns={'Date': 'Time'}).set_index('Time')
    new_data_fil = new_data[new_data.index > max(current_data.index)]
    if len(new_data_fil) == 0:
        print('No new data to add!')
        return False
    current_data_new_week_added = pd.concat([current_data, new_data_fil.iloc[[0], :]])
    symbols[1][instrument] = current_data_new_week_added
    # resave the symbols back to a pickle file
    with open(path_to_data, 'wb') as f:
        pickle.dump(symbols, f)
    return True

### create a function that will calculate the number of bars needed for a full week's worth of data depending on the given timeframe (M1, M5, M15, H1, H4, D1)
def get_bars_needed(timeframe):
    if timeframe == 'M1':
        return 60 * 24 * 5
    elif timeframe == 'M5':
        return 20 * 24 * 5
    elif timeframe == 'M15':
        return 4 * 24 * 5
    elif timeframe == 'H1':
        return 24 * 5
    elif timeframe == 'H4':
        return 6 * 5
    else:
        return 12
    
def normalize_to_range(x, x_min, x_max):
    m = (2 - 0) / (x_max - x_min)
    b = 2
    y = m * (x - x_min)
    return y

def my_get_modified_volume(env, symbol: str, volume: float) -> float:
    si = env.simulator.symbols_info[symbol]
    v = abs(volume)
    v = normalize_to_range(v, si.volume_min, 100)
    v = np.clip(v, si.volume_min, si.volume_max)
    v = round(v / si.volume_step) * si.volume_step
    return v

def find_key_by_value(dictionary, value_to_find):
    for key, value in dictionary.items():
        if value == value_to_find:
            return key
    return None 