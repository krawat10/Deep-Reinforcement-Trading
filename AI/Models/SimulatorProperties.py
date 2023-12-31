import json
import os
from pathlib import Path
from datetime import datetime, date, timedelta

from AI.Models.DQNAgentType import DQNAgentType
from AI.Models.DQNProperties import DQNProperties
from AI.Models.DecisionMakingPolicy import DecisionMakingPolicy
from AI.Models.StockProperties import StockProperties


class SimulatorProperties:
    def __init__(self, ticker: str, start_date: str, end_date: str, max_episodes: int, trading_days: int,
                 dqn_agent: DQNProperties, path: str, leverage=1):
        self.trading_days = trading_days
        self.max_episodes = max_episodes
        self.stock = StockProperties(ticker, start_date, end_date, leverage)
        self.path = path
        self.agent = dqn_agent

    def save_config(self):
        os.makedirs(self.path)
        with open(f'{self.path}/config.json', 'w') as outfile:
            outfile.write(self.toJSON())

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)

    def get_path(self) -> Path:
        return Path(self.path)

    @staticmethod
    def from_json(json_str):
        json_data = json.loads(json_str)

        stock = json_data.get('stock')
        return SimulatorProperties(
            ticker=stock.get('ticker'),
            start_date=stock.get('start_date'),
            end_date=stock.get('end_date'),
            max_episodes=json_data.get('max_episodes'),
            trading_days=json_data.get('trading_days'),
            dqn_agent=DQNProperties.from_dictionary(json_data.get('agent')),
            path=json_data.get('path'),
            leverage=stock.get('leverage', 1)  # Default to 1 if 'leverage' is not present
        )


class SimulatorPropertiesBuilder:
    _ticker: str
    _start_date = '2000-03-19'
    _end_date = '2020-12-31'
    _max_episodes = 1000
    _path: str = ''
    _dqn_agent = DQNProperties()
    _trading_days = 0
    _leverage = 1

    def create(self) -> SimulatorProperties:
        return SimulatorProperties(self._ticker, self._start_date, self._end_date, self._max_episodes,
                                   self._trading_days, self._dqn_agent, self._path, self._leverage)

    def set_ticker(self, ticker):
        self._ticker = ticker

        if self._path == '':
            self._path = f'results_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    def set_dqn_type(self, dqn_type: DQNAgentType):
        self._dqn_agent.dqn_type = dqn_type

    def set_leverage(self, leverage: int):
        self._leverage = leverage

    def set_date_range(self, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date

    def set_max_episodes(self, max_episodes):
        self._max_episodes = max_episodes

    def set_decision_making_policy(self, policy: DecisionMakingPolicy):
        self._dqn_agent.decision_making_policy = policy

    def set_trading_days(self, trading_days):
        self._trading_days = trading_days

    def set_today_trading_day(self):
        today = date.today().strftime("%Y-%m-%d")
        year_ago = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")

        self._trading_days = 1
        self.set_date_range(year_ago, today)

    def set_path_and_load(self, path):
        with open(f'{path}/config.json', 'r') as infile:
            json_str = infile.read()
        saved_config: SimulatorProperties = SimulatorProperties.from_json(json_str)

        self._dqn_agent.dqn_type = DQNAgentType(saved_config.agent.dqn_type)
        self._dqn_agent.decision_making_policy = DecisionMakingPolicy(saved_config.agent.decision_making_policy)
        self._ticker = saved_config.stock.ticker
        self._leverage = saved_config.stock.leverage
        self._path = path

    def set_postfix(self, postfix):
        self._path = f'results_{self._ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{postfix}'
