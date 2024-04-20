# -*- coding: utf-8 -*-

'''
This script is meant as example how to use the Pytrader_API in live trading.
The logic is a simple crossing of two sma averages.
'''


import time
import pandas as pd
#import talib as ta
import numpy as np
import pandas as pd
import configparser
from datetime import datetime
import pytz
from Pytrader_API_V3_02a import Pytrader_API
import sys
sys.path.append("C:/Users/WilliamFetzner/Documents/Trading/PyTrader-python-mt4-mt5-trading-api-connector-drag-n-drop")
from Pytrader_API_V3_02a import Pytrader_API
sys.path.append("C:/Users/WilliamFetzner/Documents/Trading/PyTrader-python-mt4-mt5-trading-api-connector-drag-n-drop/strategies/utils")
from LogHelper import Logger                              # for logging events

log = Logger()
log.configure()

# settings
timeframe = 'M5'
instrument = 'EURUSD'
server_IP = '127.0.0.1'
server_port = 1122  # check port number

###### input parameters ######
## Parameters for model:

## Order Parameters
volume = 0.01
trades_in_runway = 25
slippage = 5
Start_Hour = 1
End_Hour = 22
magicnumber = 2772
SL_in_pips = 20
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
News_Trading_Allowed = True
acct_protection, swap_protection_long, swap_protection_short = False, False, False

## Other Parameters ##
date_value_last_bar = 0 
number_of_bars = 100                 

#   Create simple lookup table, for the demo api only the following instruments can be traded
brokerInstrumentsLookup = {
    'EURUSD': 'EURUSD',
    'AUDCHF': 'AUDCHF',
    'NZDCHF': 'NZDCHF',
    'GBPNZD': 'GBPNZD',
    'USDCAD': 'USDCAD'}

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

brokerInstrumentsLookup = config_instruments(config, "ICMarkets")


def get_current_equity_balance():
    # Get current equity and balance
    DynamicInfo = MT.Get_dynamic_account_info()
    for prop in DynamicInfo:
        if prop == 'Account equity': 
            current_acct_equity = DynamicInfo[prop]
        if prop == 'Account balance':
            current_acct_balance = DynamicInfo[prop]
    return current_acct_equity, current_acct_balance

# Define pytrader API
MT = Pytrader_API()

connection = MT.Connect(server_IP, server_port, brokerInstrumentsLookup)
forever = True
if (connection == True):
    log.debug('Strategy started')
    while(forever):
        current_account_equity, current_account_balance = get_current_equity_balance()
        # if the current time is 0:00:00 then reset the daily account balance/equity
        ServerTime = MT.Get_broker_server_time()
        if ServerTime.hour == 0 and ServerTime.minute == 0 and ServerTime.second == 0:
            account_equity_at_last_close, account_balance_at_last_close = get_current_equity_balance()
            print(f'Account equity at close: {account_equity_at_last_close}, Account balance at close: {account_balance_at_last_close}')
            # reset acct_protection, swap_protection_long, swap_protection_short
            acct_protection, swap_protection_long, swap_protection_short = False, False, False
        
        AcctBalDrawdown = (account_balance_at_last_close - current_account_balance) / account_balance_at_last_close
        AcctEquityDrawdown = (account_equity_at_last_close - current_account_equity) / account_equity_at_last_close

        InstrumentInfo = MT.Get_instrument_info(instrument=instrument)
        if InstrumentInfo is not None:
            for prop in InstrumentInfo:
                if prop == 'swap_long':
                    swap_long = InstrumentInfo[prop]
                    if swap_long < 0:
                        swap_protection_long = True
                if prop == 'swap_short':
                    swap_short = InstrumentInfo[prop]
                    if swap_short < 0:
                        swap_protection_short = True


        # retrieve open positions
        positions_df = MT.Get_all_open_positions()
        # if open positions, check for closing, if SL and/or TP are defined.
        # using hidden SL/TP
        # first need actual bar info
        actual_bar_info = MT.Get_actual_bar_info(instrument=instrument, timeframe=MT.get_timeframe_value(timeframe))
        if (len(positions_df) > 0):
            for position in positions_df.itertuples():

                # account protection if statements
                if (position.instrument == instrument and position.magic_number == magicnumber and (AcctBalDrawdown > max_Daily_Drawdown_Perc or AcctEquityDrawdown > max_Daily_Drawdown_Perc)):
                    acct_protection = True
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug('trade with ticket ' + str(position.ticket) + ' closed due to daily drawdown protection')

                if (position.instrument == instrument and position.magic_number == magicnumber and (Max_Payout_Bool == True and (current_account_balance - Initial_Acct_Size) > Max_Payout_Amt)):
                    acct_protection = True
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug('trade with ticket ' + str(position.ticket) + ' closed due to max payout - Request payout now! Yay!')
                # close positions to not allow weekend holds
                if (position.instrument == instrument and position.magic_number == magicnumber and (ServerTime.hour == 23 & ServerTime.day == 5)):
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug('trade with ticket ' + str(position.ticket) + ' closed for no weekend hold')

                # close negative swap positions
                if (position.instrument == instrument and position.magic_number == magicnumber and ((swap_protection_long == True and position.position_type == 'buy') or
                                                                                                     (swap_protection_short == True and position.position_type == 'sell'))):
                    # close the position
                    MT.Close_position_by_ticket(ticket=position.ticket)
                    log.debug(f'trade with {position.position_type} ticket {position.ticket} closed due to negative swap protection')

                # if (position.instrument == instrument and position.position_type == 'buy' and position.magic_number == magicnumber):
                #     # tp = position.open_price + TP_in_pips / multiplier
                #     if (actual_bar_info['close'] > tp):
                #         # close the position
                #         MT.Close_position_by_ticket(ticket=position.ticket)
                #         log.debug('trade with ticket ' + str(position.ticket) + ' closed in profit')
                        
                elif (position.instrument == instrument and position.position_type == 'buy' and SL_in_pips > 0.0 and position.magic_number == magicnumber):
                    sl = position.open_price - SL_in_pips / multiplier
                    if (actual_bar_info['close'] < sl):
                        # close the position
                        MT.Close_position_by_ticket(ticket=position.ticket)
                        log.debug('trade with ticket ' + str(position.ticket) + ' closed with stoploss')
                
                # elif (position.instrument == instrument and position.position_type == 'sell' and position.magic_number == magicnumber):
                #     # tp = position.open_price - TP_in_pips / multiplier
                #     if (actual_bar_info['close'] < tp):
                #         # close the position
                #         MT.Close_position_by_ticket(ticket=position.ticket)
                #         log.debug('trade with ticket ' + str(position.ticket) + ' closed in profit')
                elif (position.instrument == instrument and position.position_type == 'sell' and SL_in_pips > 0.0 and position.magic_number == magicnumber):
                    sl = position.open_price + SL_in_pips / multiplier
                    if (actual_bar_info['close'] > sl):
                        # close the position
                        MT.Close_position_by_ticket(ticket=position.ticket)
                        log.debug('trade with ticket ' + str(position.ticket) + ' closed with stoploss')

        # only if we have a new bar, we want to check the conditions for opening a trade/position
        # at start check will be done immediatly
        # date values are in seconds from 1970 onwards.
        # for comparing 2 dates this is ok

        if (actual_bar_info['date'] > date_value_last_bar):
            date_value_last_bar = actual_bar_info['date']
            # new bar, so read last x bars
            bars = MT.Get_last_x_bars_from_now(instrument=instrument, timeframe=MT.get_timeframe_value(timeframe), nbrofbars=number_of_bars)
            # convert to dataframe
            df = pd.DataFrame(bars)
            df.rename(columns = {'tick_volume':'volume'}, inplace = True)
            df['date'] = pd.to_datetime(df['date'], unit='s')

            index = len(df) - 2
            # conditions will be checked on bar [index] and [index-1]
            if (df['sma_1'][index] > df['sma_2'][index] and df['sma_1'][index-1] < df['sma_2'][index-1]):           # buy condition
                
                buy_OK = MT.Open_order(instrument=instrument,
                                        ordertype='buy',
                                        volume = volume,
                                        openprice=0.0,
                                        slippage = slippage,
                                        magicnumber = magicnumber,
                                        stoploss=0.0,
                                        takeprofit=0.0,
                                        comment='strategy_1')  

                if (buy_OK > 0):
                    log.debug('Buy trade opened')
                    # check if not a sell position is active, if yes close this sell position 
                    for position in positions_df.itertuples():
                        if (position.instrument== instrument and position.position_type== 'sell' and position.magic_number == magicnumber):
                            # close
                            close_OK = MT.Close_position_by_ticket(ticket=position.ticket) 
                            log.debug('closed sell trade due to cross and opening buy trade') 
            
            if (df['sma_1'][index] < df['sma_2'][index] and df['sma_1'][index-1] > df['sma_2'][index-1]):           # sell condition
                
                sell_OK = MT.Open_order(instrument=instrument,
                                        ordertype='sell',
                                        volume = volume,
                                        openprice=0.0,
                                        slippage = slippage,
                                        magicnumber = magicnumber,
                                        stoploss=0.0,
                                        takeprofit=0.0,
                                        comment='strategy_1')  

                if (sell_OK > 0):
                    log.debug('Sell trade opened')
                    # check if not a buy position is active, if yes close this buy position 
                    for position in positions_df.itertuples():
                        if (position.instrument == instrument and position.position_type == 'buy' and position.magic_number == magicnumber):
                            # close
                            close_OK = MT.Close_position_by_ticket(ticket=position.ticket)
                            log.debug('closed buy trade due to cross and opening sell trade')

        # wait 2 seconds
        time.sleep(2)

        # check if still connected to MT terminal
        still_connected = MT.Check_connection()
        if (still_connected == False):
            forever = False
            print('Loop stopped')
            log.debug('Loop stopped')
