# Algorithmic Trading and Machine Learning Project

## Overview

This repository contains a collection of tools, strategies, and implementations for algorithmic trading using machine learning techniques. The project focuses on developing and testing automated trading systems for EURUSD.

## Purpose of this Repository

I was searching for a way to trade algorithmically using different strategies within Forex trading, specifically EURUSD. You can utilize anything that you would like from this repo, and gain any nuggets that you would like from within. I do not guarantee the functionality of any of the python files, jupyter notebooks, or any other files, due to updates to dependent libraries. *(Beware, gym-mtsim requires pandas version 2.0.3)* I am not a professional trader, and this is a personal project that I worked on in my free time. If you would like to understand the process that I went through, feel free to read the -Process- section below.

## Key Features

- Implementation of various trading models using machine learning algorithms
- Integration with MetaTrader platform for live trading
- Backtesting framework for evaluating strategy performance
- Data visualization tools using matplotlib and mplfinance
- Custom indicators and technical analysis implementations

## Technologies Used

- Python as the primary development language
- Gym-mtsim/gym-anytrading framework for financial reinforcement learning environments
- Genetic Algorithms/hyperopt to optimize parameters
- Matplotlib and mplfinance for data visualization
- Pandas, polars, and NumPy for data manipulation and analysis
- NIXTLA for time-series analysis and predictions
- XGBoost for Binary classification of trading performance
- MetaTrader API for integration with MT4 platform

## Process

1. Indicator Backtesting:

    a) Impulse-MACD, PSAR, and SMI - Impulse_PSAR_nb.ipynb, optimization_Impulse_PSAR_nb.ipynb
I first went through the process of trying to automate and backtest a few indicators that I had found fairly successful on EURUSD. Specifically, I looked into PSAR, impulse-MACD, and SMI (Smoothing Momentum Index). I was using the confluence of PSAR and impulse-MACD as the indicators for the entrance to a trade, and SMI as the indicator for the exit. At the time, I hadn't found the PyTrader-python-mt4-mt5-trading-api-connector-drag-n-drop repo to allow me to trade with the MT4 platform using python, so I created a MQL4 script and a bot that would do the trading with PSAR, impulse-MACD, and SMI. While I was waiting for results from the first week of testing the bot, I created a Genetic Algorithm to optimize the parameters for the indicators. The genetic algorithm worked great, except that it majorly overfit the data, and started suggesting wild parameters that worked great for the validation data, but then failed miserably on the test data. I was able to combat the overfitting issue later on during the optimization of the Double-SMA strategy, which comes next. At the time that I was finding the overfitting issue, the results from the week of trading came back, and the bot and backtesting didn't align due to differences in the calculations of the indicators and the data collection between MT4 and my code in python. 

    b) Double-SMA - double3_sma_nb.ipynb, optimization_dsma_nb.ipynb
So I went simpler. I started going through the process of testing a double-SMA strategy, which is a simple moving average crossover strategy. It involved using a 3 bar simple moving average, then creating a second 9 bar moving average based on the first SMA line. I was able to get the backtesting to work, but the results were not very promising. So I also brought in using Renko charts with double-sma. However, there were several challenges with this, as I still hadn't found the MT4 connector, so trying to align my backtesting python data with my MT4 bot was a challenge. Especially given that MT4 doesn't have a Renko chart feature, so I was relying on a Renko chart EA, and all of this depended on the starting point of the Renko bar. Then during optimization of the parameters, one of the parameters I used was the Renko bar size, but this just minimizes it to the lowest possible Renko bar size, which is 0.1 pips. This works fine on backtesting data, where all the times are known, but in live trading environment, you have drastic swings and the EA renko just couldn't handle these fluctuations, and the bot couldn't trade the effective way that it showed in the backtesting. Plus, when you get that low, you have to deal with spread, which I wasn't accounting for at the beginning of the backtesting. When I brought backtesting in, I found that the backtesting was not successful at all, and the optimizations went back to overfitting the data, and just didn't work well. 

2. Reinforcement Learning: -hyperparameter_search.ipynb, hyperparameter_analysis.ipynb, saving_tested_model.ipynb, nixtla_exploration.ipynb, optimization_regression.ipynb, Ensemble_RL_SARIMA.ipynb

After the forays into the backtesting the indicators, I was able to find the MT4 connector! Horray! This opened up a whole new world of possibilities. Especially given that I can backtest and have the real thing go live in the same environment. Plus, this was also the time that I found out about the gym-anytrading and gym-mtsim repos. And given that I am a data scientist, I decided to give RL a try. So I started piggy-backing off of the amazing work of first gym-anytrading, then upgraded to their gym-mtsim environment for reinforcement learning. Instead of using the GA for optimizing the parameters I used hyperopt. 

  a) GYM-MTSIM - Base
        I did a considerable amount of work here to try to optimize the parameters using Gym-mtsim simulations and then I was able to connect it to the MT4 platform using the MT4 EA python to MT4 platform. This worked out fairly well, and I was able to run this for several weeks, as you can see in the orders folder. However, the main issue is that when you are performing an anlysis with an RL model is that with the same hyperparameters it creates a distribution, over thousands of runs, many of them are successful, but it is a distribution, and they average to be mostly profitable, but not always. I went through the process of training the model on past data up to a week prior to the current date, so I would train for 50_000 bars of hourly data, then validate the parameters on the last week of data, then use the best performing model and save it to then be used on the live environment the following week. I had some positive weeks and some negative weeks, but nothing consistent so I went searching for an additional way to improve my results. 

   b) GYM-MTSIM - Ensemble
        I started looking into Ensemble methods combining the RL with something else. \n
       
  i) GYM-MTSIM - Nixtla
            Nixtla has a variety of amazing time-series analysis tools that I began experimenting with. I landed on the Random Walk with Drift and basic linear regression model using their AutoLinearRegression function. I then added this to the RL model, and it didn't seem to help much. I then started trying to create a strategy around the predictions provided by the RWD and LR, but couldn't land on one that was successful

   ii) GYM-MTSIM - XGBoostClassifier
            I then added an XGBoostClassifier to the RL, that would train on all of the orders that the RL model made during training and then use that information to predict how well the RL was doing at creating profitable trades, but predicting on the trade coming up whether it would be a successful or not successful trade based on how the RL model did during training. If the Classification model predicted a bad trade, it would rewrite the action to not open the trade. This did seem to perform slightly better. However, I did not get a chance to implement this, and I do not foresee sinking more time into this project, although it is a possible opportunity in the future to bring this to implementation. 

## Contributing

Contributions are welcome! Please submit pull requests with clear descriptions of changes made.

## Acknowledgments

- A huge shoutout goes to those repos that are forked within this repository especially Gym_mtsim/gym-anytrading which laid the foundation for the reinforcement learning environments.
- Also a huge shoutout to PyTrader-python-mt4-mt5-trading-api-connector-drag-n-drop which allowed me to trade with my reinforcement learning bot that was coded in python on Metatrader4 which does not have an API for trading with other programming languages like python. 
- And Nixtla for providing an amazing repository of time-series analysis tools. 
