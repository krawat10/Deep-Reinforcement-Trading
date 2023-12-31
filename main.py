import os
from datetime import date, timedelta
from enum import Enum

from alpaca.trading import OrderSide, TimeInForce
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest
from Plotter import Plotter
from Simulator import Simulator
from Models.SimulatorProperties import SimulatorPropertiesBuilder
from Models.DQNAgentType import DQNAgentType
from Models.DecisionMakingPolicy import DecisionMakingPolicy
from TradingClient import TradingManager


class Mode(Enum):
    RUN_NEW_SIMULATION = 1
    RUN_EXISTING_NETWORK = 2
    RUN_SINGLE_DAY_PREDICTION = 3
    PLOT_RESULTS = 4


# mode = Mode.RUN_EXISTING_NETWORK
mode = Mode.RUN_NEW_SIMULATION
#mode = Mode.RUN_SINGLE_DAY_PREDICTION
# mode = Mode.PLOT_HISTOGRAMS

shutdown_after_simulation = False  # Turn off after simulation (it takes some time)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
if mode == Mode.RUN_NEW_SIMULATION:
    builder = SimulatorPropertiesBuilder()
    builder.set_date_range('1999-01-01', '2020-01-01')
    builder.set_max_episodes(1200)
    builder.set_ticker('INTC')
    builder.set_dqn_type(DQNAgentType.DDQN)
    builder.set_decision_making_policy(DecisionMakingPolicy.EPSILON_GREEDY_POLICY)
    builder.set_postfix('250_2003_2020_1200_1024_1024_DDQN_EPSILON_GREEDY_POLICY_5X')
    builder.set_trading_days(250)
    builder.set_leverage(5)
    properties = builder.create()
    properties.save_config()
    Simulator().run_new(properties)

elif mode == Mode.PLOT_RESULTS:
    Plotter.plot_multiple_nav([
        'results_INTC_20221123_163018_250_2003_2020_250_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221123_182933_250_2003_2020_500_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221124_115511_250_2003_2020_1000_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221122_200205_250_2003_2020_1500_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221125_234052_250_2003_2020_2000_DDQN_EPSILON_GREEDY_POLICY'
    ])
    Plotter.plot_multiple_detailed_histograms([
        'results_INTC_20221123_163018_250_2003_2020_250_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221123_182933_250_2003_2020_500_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221124_115511_250_2003_2020_1000_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221122_200205_250_2003_2020_1500_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221125_234052_250_2003_2020_2000_DDQN_EPSILON_GREEDY_POLICY'
    ])
    Plotter.plot_multiple_histograms([
        'results_INTC_20221123_163018_250_2003_2020_250_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221123_182933_250_2003_2020_500_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221124_115511_250_2003_2020_1000_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221122_200205_250_2003_2020_1500_DDQN_EPSILON_GREEDY_POLICY',
        'results_INTC_20221125_234052_250_2003_2020_2000_DDQN_EPSILON_GREEDY_POLICY'
    ])

elif mode == Mode.RUN_EXISTING_NETWORK:
    builder = SimulatorPropertiesBuilder()
    builder.set_date_range('2020-01-01', '2022-06-01')
    builder.set_path_and_load('results_INTC_20221211_022531_250_2003_2020_1200_DDQN_EPSILON_GREEDY_POLICY_5X')
    builder.set_trading_days(604)
    properties = builder.create()
    Simulator().run_existing(properties)

elif mode == Mode.RUN_SINGLE_DAY_PREDICTION:
    api = TradingManager()
    builder = SimulatorPropertiesBuilder()
    builder.set_path_and_load('results_INTC_20221125_234052_250_2003_2020_2000_DDQN_EPSILON_GREEDY_POLICY')
    builder.set_today_trading_day()
    properties = builder.create()
    action = Simulator().run_single_day(properties)  # Prediction for the next day (to make real bets)
    if action > 0:
        print('LONG')
        api.submit_long('INTC', 1)
    elif action == 0:
        print('CASH')
        api.close_all_positions()
    else:
        print('SHORT')
        api.submit_short('INTC', 1)


if shutdown_after_simulation:
    os.system("shutdown /s /t 1")

# Visualization
