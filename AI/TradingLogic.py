import numpy as np
import pandas as pd


class TradingLogic:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps, leverage):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        self.leverage = leverage

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.dates = np.empty(self.steps, dtype='datetime64[s]')
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.start_positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.dates.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.start_positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return, date):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """


        self.dates[self.step] = date
        start_position = self.positions[max(0, self.step - 1)]
        end_position = action - self.leverage
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        # short, neutral, long
        # end_position = action * 2 - 1  # short, long
        # end_position = action # neutral, long
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.start_positions[self.step] = start_position
        self.trades[self.step] = n_trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[self.step]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])

        info = {'reward': reward,
                'nav': self.navs[self.step],
                'costs': self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self) -> pd.DataFrame:
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action': self.actions,  # current action
                             'nav': self.navs,  # starting Net Asset Value (NAV)
                             'market_nav': self.market_navs,
                             'market_return': self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position': self.positions,  # eod position
                             'start_position': self.start_positions,  # eod position
                             'cost': self.costs,  # eod costs
                             'trade': self.trades,
                             'dates': self.dates})  # eod trade)
