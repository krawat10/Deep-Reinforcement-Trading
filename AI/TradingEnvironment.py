import gym
from gym.utils import seeding

from AI.DataSource import DataSource
from AI.Models.StockProperties import StockProperties
from AI.TradingLogic import TradingLogic
import numpy as np

class TradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human'], 'observation_space.names': []}

    def __init__(self,
                 properties: StockProperties,
                 data_source: DataSource):
        self.end_date = properties.end_date
        self.start_date = properties.start_date
        self.trading_cost_bps = properties.trading_cost_bps
        self.time_cost_bps = properties.time_cost_bps
        self.ticker = properties.ticker
        self.data_source = data_source
        self.leverage = properties.leverage
        self.simulator = TradingLogic(steps=self.data_source.trading_days,
                                      trading_cost_bps=self.trading_cost_bps,
                                      time_cost_bps=self.time_cost_bps,
                                      leverage=properties.leverage)

        self.action_space = gym.spaces.Discrete(properties.leverage * 2 + 1)  # 2 x SHORTS, 2 x LONGS + NEUTRAL
        self.observation_space = gym.spaces.Box(self.data_source.min_values.values, self.data_source.max_values.values)
        self.metadata['observation_space.names'] = list(self.data_source.data.columns.values)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, date, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                market_return=observation[0],
                                                date=date)
        return observation, reward, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass
