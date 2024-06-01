# -*- coding: utf-8 -*-
# to begin running the terminal & cd C:/Users/Administrator/AppData/Local/Programs/Python/Python312/
# .\python C:\Users\Administrator\Documents\Trading\PyTrader_RL_MSolutions.py


'''
This script is meant as example how to use the Pytrader_API in live trading.
The logic is a simple crossing of two sma averages.

'''
import numpy as np
import pandas as pd
import gymnasium as gym
import pickle
import gym_mtsim
from gym_mtsim_forked.gym_mtsim.data import FOREX_DATA_PATH_PRODUCTION_MS, FOREX_DATA_PATH, MODEL_PATH
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import time
# import torch
#import talib as ta
import configparser
from datetime import datetime
# import pytz
# reimport this module
import fx_rl
from Pytrader_API_V3_02a import Pytrader_API
from LogHelper import Logger                              # for logging events
log = Logger()
log.configure()
# settings
timeframe = 'H1'
instrument = 'EURUSD'
server_IP = '127.0.0.1'
server_port = 2981  # check port number
seed = 2024
# IG Account info
# 744127
# w8gmsxe

###### input parameters ######
## Order Parameters
volume = 0.5
trades_in_runway = 25
slippage = 5
Start_Hour = 0
End_Hour = 23
magicnumber = 2772
SL_in_pips = 10
TP_in_pips = 10
multiplier = 10_000  # multiplier for calculating SL and TP, for JPY pairs should have the value of 100
if instrument.find('JPY') >= 0:
    multiplier = 100.0  

##  Account Settings ##
max_Daily_Drawdown_Perc = 0.03
max_Total_Drawdown_Amt = 184_000
Max_Payout_Bool = False
Max_Payout_Amt = 12_000
Initial_Acct_Size = 200_000
News_Trading_Allowed = False
acct_protection, swap_protection_long, swap_protection_short = False, False, False

## Other Parameters ##
date_value_last_bar = 0 
number_of_bars = 100 
more_trades = True                

# Define pytrader API
MT = Pytrader_API()

def config_instruments(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            option = option.upper()
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except BaseException:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


# Read in config
CONFIG_FILE = "Instrument.conf"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)

# brokerInstrumentsLookup = config_instruments(config, "MSolutions")
brokerInstrumentsLookup = {
    'EURUSD': 'EURUSD.i',
    'AUDCHF': 'AUDCHF.i',
    'NZDCHF': 'NZDCHF.i',
    'GBPNZD': 'GBPNZD.i',
    'USDCAD': 'USDCAD.i'}
connection = MT.Connect(server_IP, server_port, brokerInstrumentsLookup)
print(connection)
IsAlive = MT.connected
print(IsAlive)
# time.sleep(2)

def get_current_equity_balance():
    # Get current equity and balance
    DynamicInfo = MT.Get_dynamic_account_info()
    for prop in DynamicInfo:
        if prop == 'equity': 
            current_acct_equity = DynamicInfo[prop]
        if prop == 'balance':
            current_acct_balance = DynamicInfo[prop]
    return current_acct_equity, current_acct_balance

# class MyMtEnv(gym_mtsim.MtEnv):
    # _get_modified_volume = fx_rl.my_get_modified_volume
    # _get_prices = fx_rl.my_get_prices

# load in the model to use for the week
sim_train = gym_mtsim.MtSimulator(
    unit='USD',
    balance=200_000.,
    leverage=100.,
    stop_out_level=0.2,
    hedge=True,
    symbols_filename=FOREX_DATA_PATH
)
sim_training_fee = lambda symbol: {
    instrument: max(0., np.random.normal(0.0001, 0.00003))
}[symbol]

# time how long this code takes
# start = time.time()
train_env = gym_mtsim.MtEnv(
    original_simulator=sim_train,
    trading_symbols=[instrument],
    window_size = 10,
    # time_points=list(training_index_slice),
    hold_threshold=0.5,
    close_threshold=0.5,
    fee=sim_training_fee,
    symbol_max_orders=2
    # multiprocessing_processes=1
)
# end = time.time()
# print(f'Environment creation time: {end - start} seconds')
# time.sleep(120)
model = PPO.load(MODEL_PATH, env=train_env)

# initialize model and environment
ServerTime = MT.Get_broker_server_time()
print(ServerTime)
initial_account_equity, initial_account_balance = get_current_equity_balance()
print(f'Initial account equity: {initial_account_equity}, Initial account balance: {initial_account_balance}')

initial_spread = MT.Get_last_tick_info(instrument=instrument)['spread'] / (multiplier * 10)
print(f'Initial spread: {initial_spread}')

InstrumentInfo = MT.Get_instrument_info(instrument=instrument)
if InstrumentInfo is not None:
    for prop in InstrumentInfo:
        if prop == 'swap_long':
            swap_long = InstrumentInfo[prop]
            if swap_long < 0:
                swap_protection_long = True
                # print(f'swap_long: {swap_long}')
        if prop == 'swap_short':
            swap_short = InstrumentInfo[prop]
            if swap_short < 0:
                swap_protection_short = True
                # print(f'swap_short: {swap_short}')

# get the last week's worth of data for the production environment
# bars = MT.Get_last_x_bars_from_now(instrument=instrument, timeframe=MT.get_timeframe_value(timeframe), nbrofbars=fx_rl.get_bars_needed(timeframe))
bars = MT.Get_last_x_bars_from_now(instrument=instrument, timeframe=MT.get_timeframe_value(timeframe), nbrofbars=120)
time.sleep(2)
# convert to dataframe
df = pd.DataFrame(bars)
# df.rename(columns = {'tick_volume':'volume'})#, inplace = True)
df['date'] = pd.to_datetime(df['date'], unit='s')
df.columns = [col.capitalize() for col in df.columns]
df = df.rename(columns={'Date': 'Time'}).set_index('Time')
with open(FOREX_DATA_PATH_PRODUCTION_MS, 'rb') as f:
    import pickle
    symbols_new = pickle.load(f)
# replace the data in the pickle file with the new data
symbols_new[1][instrument] = df
with open(FOREX_DATA_PATH_PRODUCTION_MS, 'wb') as f:
    pickle.dump(symbols_new, f)

sim_production = gym_mtsim.MtSimulator(
    unit='USD',
    balance=initial_account_balance,
    leverage=100.,
    stop_out_level=0.2,
    hedge=True,
    symbols_filename=FOREX_DATA_PATH_PRODUCTION_MS
)

class MyMtEnv2(gym_mtsim.MtEnv):
    _get_modified_volume = fx_rl.my_get_modified_volume
    # _get_prices = fx_rl.my_get_prices
# start_2 = time.time()
env_production = MyMtEnv2(
    original_simulator=sim_production,
    trading_symbols=[instrument],
    window_size = 10,
    # time_points=list(training_index_slice),
    hold_threshold=0.5,
    close_threshold=0.5,
    fee=sim_training_fee,
    symbol_max_orders=2
    # multiprocessing_processes=1
)
# end_2 = time.time()
# print(f'Production Environment creation time: {end_2 - start_2} seconds')
# time.sleep(2)
obs_production, info_production = env_production.reset(seed=seed)
# model.set_env(env_production) # do I need this? I didn't even know this existed
done_production = False
stop = False
# run through the environment up to the current week and stop with the model
iteration = 0
diff_balances = -1 
while diff_balances < 0:
    while not done_production:
        action_pre_production, _states = model.predict(obs_production)
        if (len(env_production.time_points) - 2) == env_production._current_tick:
            stop = True
            # print('closing all orders')
            # if there are any open orders, we need to go ahead and close them
            for o in range(len(env_production.simulator.symbol_orders('EURUSD'))):
                # print(f'closing order {o}')
                action_pre_production[o] = 0.99
                # make sure the model doesn't place another one
            action_pre_production[-2] = 0.99
        obs_production, reward_production, terminated_production, truncated_production, info_production = env_production.step(action_pre_production)
        if stop:
            truncated_production = True
        done_production = terminated_production or truncated_production
        if done_production:
            break
    end_balance_pre_prod = env_production.render()['orders']['Exit Balance'].values[0]
    starting_balance_pre_prod = env_production.render()['orders']['Exit Balance'].values[-1]
    diff_balances = end_balance_pre_prod - starting_balance_pre_prod
    iteration += 1
    if iteration > 100:
        print('Too many iterations')
        break
print(f'iterations: {iteration}, positive_balance: {diff_balances}')
#  env_production.render()['orders']
# dictionary to store the trade_id and the corresponding ticket number
trade_id_conversion = {}
forever = True
count_of_times_getting_new_data = 0
if (connection == True):
    log.debug('Strategy started')
    while(forever):
        current_account_equity, current_account_balance = get_current_equity_balance()
        # if the current time is 0:00:00 then reset the daily account balance/equity
        ServerTime = MT.Get_broker_server_time()
        # if it is midnight or if account_equity_at_last_close and account_balance_at_last_close are not defined
        if (ServerTime.hour == 0 and ServerTime.minute == 0 and ServerTime.second == 0) | ((not 'account_equity_at_last_close' in locals()) | (not 'account_balance_at_last_close' in locals())):
            account_equity_at_last_close, account_balance_at_last_close = get_current_equity_balance()
            print(f'Account equity at close: {account_equity_at_last_close}, Account balance at close: {account_balance_at_last_close}')
            # reset acct_protection, swap_protection_long, swap_protection_short
            acct_protection = False
        
        AcctBalDrawdown = (account_balance_at_last_close - current_account_balance) / account_balance_at_last_close
        AcctEquityDrawdown = (account_equity_at_last_close - current_account_equity) / account_equity_at_last_close

        # retrieve open positions
        positions_df = MT.Get_all_open_positions()
        # positions_df
        # if open positions, check for closing, if SL and/or TP are defined.
        # using hidden SL/TP
        # first need actual bar info
        actual_bar_info = MT.Get_actual_bar_info(instrument=instrument, timeframe=MT.get_timeframe_value(timeframe))
        if (len(positions_df) > 0):
            for position in positions_df.itertuples():

                # account protection if statements
                if (position.instrument == (instrument + '.i') and position.magic_number == magicnumber and (AcctBalDrawdown > max_Daily_Drawdown_Perc or AcctEquityDrawdown > max_Daily_Drawdown_Perc)):
                    acct_protection = True
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug('trade with ticket ' + str(position.ticket) + ' closed due to daily drawdown protection')
                    print(f'trade with ticket ' + str(position.ticket) + ' closed due to daily drawdown protection')

                if (position.instrument == (instrument + '.i') and position.magic_number == magicnumber and (Max_Payout_Bool == True and (current_account_balance - Initial_Acct_Size) > Max_Payout_Amt)):
                    acct_protection = True
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug('trade with ticket ' + str(position.ticket) + ' closed due to max payout - Request payout now! Yay!')
                    print(f'trade with ticket ' + str(position.ticket) + ' closed due to max payout - Request payout now! Yay!')

                # close positions to not allow weekend holds
                if (position.instrument == (instrument + '.i') and position.magic_number == magicnumber and ((ServerTime.hour == 23) & (ServerTime.weekday() == 4))):
                    acct_protection = True
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug('trade with ticket ' + str(position.ticket) + ' closed for no weekend hold')
                    print(f'trade with ticket ' + str(position.ticket) + ' closed for no weekend hold')

                # close negative swap positions
                if ((position.instrument == (instrument + '.i')) & (position.magic_number == magicnumber) & 
                    (((swap_protection_long == True) & (position.position_type == 'buy')) | ((swap_protection_short == True) & (position.position_type == 'sell'))) &
                    ((ServerTime.hour == 23) & (ServerTime.minute >= 55))
                    ):
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug(f'trade with {position.position_type} ticket {position.ticket} closed due to negative swap protection')
                    print(f'trade with {position.position_type} ticket {position.ticket} closed due to negative swap protection')

        # only if we have a new bar, we want to check the conditions for opening a trade/position
        # at start check will be done immediatly
        # date values are in seconds from 1970 onwards.
        # for comparing 2 dates this is ok

        if (actual_bar_info['date'] > date_value_last_bar) and (not acct_protection):
            more_trades = True

            ######## check if within 10 minutes of a news event ########
            if (News_Trading_Allowed == False):
                news_happening = fx_rl.get_news_from_csv(News_Trading_Allowed, ServerTime)
                # print(f'news happening = {news_happening}')
                if news_happening:
                    more_trades = False
                    # print(f'more trades = {more_trades}')
                    log.debug('No trades allowed due to news event')
                    print('No trades allowed due to news event')

            ######## check if within trading hours ########
            if ((ServerTime.hour < Start_Hour) or (ServerTime.hour > End_Hour)):
                more_trades = False
                log.debug('No trades allowed due to not being within the trading hours')
                print('No trades allowed due to not being within the trading hours')
            elif ((ServerTime.hour >= Start_Hour) and (ServerTime.hour <= End_Hour) and more_trades):
                more_trades = True
            # print(more_trades)

            ######## calculate the volume for the orders ########
            all_positions_df = MT.Get_closed_positions_within_window(date_to=ServerTime)
            current_instrument_all_positions = all_positions_df[all_positions_df['instrument'] == (instrument + '.i')]
            if (len(current_instrument_all_positions) > 0):
                # find the average volume for current_instrument_all_positions
                avg_volume = current_instrument_all_positions.loc[:, 'volume'].mean()
                max_volume = round(avg_volume * 2, 2)
                min_volume = round(avg_volume / 4, 2)
                todays_drawdown_limit = max_Daily_Drawdown_Perc * account_balance_at_last_close
                todays_drawdown_diff = current_account_balance - (current_account_balance - todays_drawdown_limit)
                max_drawdown_diff = current_account_balance - max_Total_Drawdown_Amt
                closer_drawdown_diff = min(todays_drawdown_diff, max_drawdown_diff)

                risk_per_trade = closer_drawdown_diff / trades_in_runway
                risk_volume = round(risk_per_trade / ((SL_in_pips / multiplier) * 100_000), 2)
                calculated_volume = min(max_volume, risk_volume)
                if (volume != 0.01) and (calculated_volume < volume):
                    volume = max(min_volume, calculated_volume)
                print(f'volume: {volume}')          

            date_value_last_bar = actual_bar_info['date']
            # new bar, so read last x bars
            bars = MT.Get_last_x_bars_from_now(instrument=instrument, timeframe=MT.get_timeframe_value(timeframe), nbrofbars=10)
            # convert to dataframe
            df_new = pd.DataFrame(bars)
            # df.rename(columns = {'tick_volume':'volume'})#, inplace = True)
            df_new['date'] = pd.to_datetime(df_new['date'], unit='s')           

            ## add the data to the environment
            data_added = fx_rl.get_latest_data(FOREX_DATA_PATH_PRODUCTION_MS, df_new, instrument=instrument)
            if data_added:
                count_of_times_getting_new_data += 1
                sim_production.load_symbols(FOREX_DATA_PATH_PRODUCTION_MS)
                # obs_production['features'] - tell it to get observation right here - or do all the data processing beolw right here, and just say _get_observation()
                action, _states = model.predict(obs_production)
                swap_protection = False
                if ((action[-1] > 0) & (ServerTime.hour == 23)): # the last item in action is the volume, if it is positive it is a long trade
                    swap_protection = swap_protection_long
                elif ((action[-1] < 0) & (ServerTime.hour == 23)):
                    swap_protection = swap_protection_short
                # if we shouldn't place an order right now due to more_trades being False or swap_protection being True, 
                # then we set the hold_probability (action[2]) to 0.99 which is greater than the hold threshold of 0.5
                # so that the model will not simulate an order
                if ((not more_trades) | (swap_protection)):
                    action2 = 0.99
                    action[-2] = action2
                    log.debug(f'No trades allowed due to either swap {swap_protection} or more_trades {more_trades} being False')
                    print(f'No trades allowed due to either swap {swap_protection} or more_trades {more_trades} being False')
                # if a trade was closed due to the stoploss, close it for the simulator
                # get the list of all the ticket numbers for all the closed positions from all_positions_df
                closed_tickets = all_positions_df.ticket.values
                if (len(closed_tickets) > 0):
                    for ticket in closed_tickets:
                        if (ticket in trade_id_conversion.values()):
                            # get the key from the value within the trade_id_conversion dictionary
                            key = fx_rl.find_key_by_value(trade_id_conversion, ticket)
                            # filter current_orders to the key
                            orders_closed_by_sl = current_orders[current_orders['Id'] == key]
                            if (orders_closed_by_sl.iloc[0, -1] == False):
                                # Grab the entry price
                                orders_table_entry_price = orders_closed_by_sl['Entry Price']
                                # find the index within the 'orders' of obs_production
                                for idx, order in enumerate(obs_production['orders'][0]): # this will break when using more than one instrument
                                    if order[0] == orders_table_entry_price.values[0]:
                                        action[idx] = 0.99
                        
                env_production.time_points = list(sim_production.symbols_data[instrument].index)
                env_production.simulator.symbols_data = sim_production.symbols_data
                env_production.simulator.current_time = env_production.time_points[env_production._current_tick]
                env_production._end_tick = len(env_production.time_points) - 1
                print(env_production._current_tick)
                env_production.prices = env_production._get_prices()
                env_production.signal_features = env_production._process_data()
                env_production.features_shape = (env_production.window_size, env_production.signal_features.shape[1])
                env_production.fee = MT.Get_last_tick_info(instrument=instrument)['spread'] / (multiplier * 10)
                obs_production, reward_production, terminated_production, truncated_production, info_production = env_production.step(action)
                current_orders = env_production.render()['orders']
                # save the current_orders to a csv file
                current_orders.to_csv('C:/Users/Administrator/Documents/Trading/current_orders_ms.csv', index=False)
                print(current_orders)
                # convert current_orders['Entry Time'] to datetime
                current_orders['Entry Time'] = pd.to_datetime(current_orders['Entry Time'])
                # find the max Entry Time
                max_entry_time = current_orders['Entry Time'].max()
                if 'H' in timeframe:
                    # get the number after the 'H' in the timeframe
                    hours = int(timeframe.split('H')[1])
                    actual_entry_time = max_entry_time + pd.Timedelta(hours=hours)
                elif 'M' in timeframe:
                    # get the number after the 'M' in the timeframe
                    minutes = int(timeframe.split('M')[1])
                    actual_entry_time = max_entry_time + pd.Timedelta(minutes=minutes)
                elif 'D' in timeframe:
                    # get the number after the 'M' in the timeframe
                    days = int(timeframe.split('D')[1])
                    actual_entry_time = max_entry_time + pd.Timedelta(days=days)
                else:
                    ValueError('Timeframe not recognized')
                # if the max Entry Time is within the last 30 seconds, then open a trade
                if ((actual_entry_time >= (ServerTime - pd.Timedelta(seconds=30))) & (actual_entry_time <= ServerTime)): 
                    # filter current_orders to the max_entry_time
                    new_order = current_orders[(current_orders['Entry Time'] == (max_entry_time)) & (current_orders['Symbol'] == instrument)]#pd.Timedelta(hours=1)
                    if count_of_times_getting_new_data < 2:
                        volume_init = 0.01
                    else:
                        volume_init = volume
                    # print(new_order)
                    order_type = new_order['Type'].values[0].lower()
                    order_OK = MT.Open_order(instrument=instrument,
                            ordertype = order_type,
                            volume = volume_init,
                            openprice=0.0,
                            slippage = slippage,
                            magicnumber = magicnumber,
                            stoploss=0.0,
                            takeprofit=0.0,
                            comment='RL_PPO_strategy')
                    # order_test = MT.Open_order(instrument='EURUSD', ordertype = 'buy', volume = 0.01, openprice=0.0, slippage = slippage, magicnumber = magicnumber,stoploss=0.0, takeprofit=0.0,comment='test') 

                    if (order_OK > 0):
                        log.debug(f'{order_type} with ticket number {order_OK} trade opened')
                        print(f'{order_type} with ticket number {order_OK} trade opened')
                        trade_id_conversion[new_order['Id'].values[0]] = order_OK
                        # convert trade_id_conversion to a dataframe
                        trade_id_conversion_df = pd.DataFrame(list(trade_id_conversion.items()), columns=['Id', 'ticket'])
                        trade_id_conversion_df.to_csv('C:/Users/Administrator/Documents/Trading/trade_id_conversion_ms.csv', index=False)
                        open_positions = MT.Get_all_open_positions()
                        # filter the open positions to just the current ticket number
                        new_order_open_price = open_positions[open_positions['ticket'] == order_OK].open_price.values[0]
                        if (order_type == 'buy'):
                            new_order_sl = new_order_open_price - (SL_in_pips / multiplier)
                        elif (order_type == 'sell'):
                            new_order_sl = new_order_open_price + (SL_in_pips / multiplier)
                        stoploss_set = MT.Set_sl_and_tp_for_position(ticket=order_OK, stoploss=new_order_sl)
                        if (stoploss_set == True):
                            log.debug(f'Stoploss set for ticket {order_OK}')
                        else:
                            log.debug(f'Error setting stoploss for ticket {order_OK}')
                    else:
                        log.debug('Error opening trade')
                        log.debug(MT.order_error)
                        log.debug(MT.order_return_message)
            
            ######## RL Model Closing conditions ########
            open_positions_close_check = MT.Get_all_open_positions()
            if (len(open_positions_close_check) > 0):
                for position in open_positions_close_check.itertuples():
                    # get the key from the value within the trade_id_conversion dictionary
                    key = fx_rl.find_key_by_value(trade_id_conversion, position.ticket)
                    # filter current_orders to the key
                    orders_to_close = current_orders[current_orders['Id'] == key]
                    if orders_to_close.iloc[0, -1] == True:
                        close_OK = MT.Close_position_by_ticket(ticket=position.ticket)
                        if (close_OK == True):
                            print(f'Closed trade with ticket {position.ticket} due to signal from model')
                            log.debug(f'Closed trade with ticket {position.ticket} due to signal from model')
                        else:
                            print(f'Error closing trade with ticket {position.ticket} even though the model signaled to close it')
                            log.debug(f'Error closing trade with ticket {position.ticket} even though the model signaled to close it')

        # wait 2 seconds
        time.sleep(2)

        # check if still connected to MT terminal
        still_connected = MT.Check_connection()
        if (still_connected == False):
            forever = False
            print('Loop stopped')
            log.debug('Loop stopped')
            break
