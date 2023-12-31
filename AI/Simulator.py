from datetime import datetime
from time import time

import gym
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from gym import Env
from gym.envs.registration import register

from AI.DQNAgent import DQNAgent
from AI.DataSource import DataSource
from AI.Models.DQNProperties import DQNProperties
from AI.Models.SimulatorProperties import SimulatorProperties
from Utils.Plotter import Plotter
from Utils.utils import format_time


class Simulator:
    # Initialize variables
    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

    # Train Agent
    start = time()
    results = []

    def run_single_day(self, properties: SimulatorProperties):
        print('Prepare libs (tensorflow etc.)')
        self.prepare_libraries()

        print('Prepare open ai env')
        trading_environment = self.prepare_open_ai(properties)

        # Do not use random learning
        properties.agent.epsilon_start = 0

        print('Prepare ddqn agent')
        ddqn_agent = self.create_dqn_agent(properties.agent, properties.get_path())

        print('Run step')
        action = self.predict_single_day(trading_environment, properties, ddqn_agent)

        print(f'Selected action: {int(action)}')

        return int(action) - properties.stock.leverage

    def run_existing(self, properties: SimulatorProperties):
        print('Prepare libs (tensorflow etc.)')
        self.prepare_libraries()

        print('Prepare open ai env')
        trading_environment = self.prepare_open_ai(properties)

        # Do not use random learning
        properties.agent.epsilon_start = 0

        print('Prepare ddqn agent')
        ddqn_agent = self.create_dqn_agent(properties.agent, properties.get_path())

        print('Run step')
        self.run_step(trading_environment, properties, ddqn_agent)

    def run_new(self, properties: SimulatorProperties):
        print('Prepare libs (tensorflow etc.)')
        self.prepare_libraries()

        print('Prepare open ai env')
        trading_environment = self.prepare_open_ai(properties)

        print('Prepare ddqn agent')
        ddqn_agent = self.create_dqn_agent(properties.agent)

        print('Run step loop')
        result = self.run_steps(trading_environment, properties, ddqn_agent)

        print('Save Results')
        self.save_results(result, properties)

    @staticmethod
    def prepare_libraries():
        # Randomize
        np.random.seed(42)
        tf.random.set_seed(42)
        sns.set_style('whitegrid')

        # # Set GPU device
        # gpu_devices = tf.config.list_physical_devices('GPU')
        # if len(gpu_devices) > 0:
        #     tf.config.experimental.set_memory_growth(gpu_devices[0], True)

        tf.keras.backend.clear_session()

    def prepare_open_ai(self, properties: SimulatorProperties) -> Env:
        # Open AI environment
        register(
            id='trading-v1',
            entry_point='AI.TradingEnvironment:TradingEnvironment',
            max_episode_steps=properties.trading_days
        )

        data_source = DataSource(trading_days=properties.trading_days,
                                 start_date=properties.stock.start_date,
                                 end_date=properties.stock.end_date,
                                 ticker=properties.stock.ticker)
        Plotter.plot_data_source(data_source, properties.get_path())

        trading_environment = gym.make('trading-v1', properties=properties.stock, data_source=data_source)
        trading_environment.seed(42)

        properties.agent.state_dim = trading_environment.observation_space.shape[0]
        properties.agent.state_names = trading_environment.metadata['observation_space.names']
        properties.agent.num_actions = trading_environment.action_space.n

        return trading_environment

    def create_dqn_agent(self, properties: DQNProperties, network_path=None):
        """
        Prepare DDQN agent which will predict next actions
        :param properties: Simulation properties
        :param network_path: Saved network path (None if not create new)
        """
        dqn = DQNAgent(state_dim=properties.state_dim,
                       num_actions=properties.num_actions,
                       learning_rate=properties.learning_rate,
                       gamma=properties.gamma,
                       epsilon_start=properties.epsilon_start,
                       epsilon_end=properties.epsilon_end,
                       epsilon_decay_steps=properties.epsilon_decay_steps,
                       epsilon_exponential_decay=properties.epsilon_exponential_decay,
                       replay_capacity=properties.replay_capacity,
                       architecture=properties.architecture,
                       l2_reg=properties.l2_reg,
                       tau=properties.tau,
                       batch_size=properties.batch_size,
                       decision_making_policy=properties.decision_making_policy,
                       dqn_type=properties.dqn_type,
                       saved_network_dir=network_path)

        print(dqn.online_network.summary())

        return dqn

    def track_results(self, episode, nav_ma_100, nav_ma_10,
                      market_nav_100, market_nav_10,
                      win_ratio, total, epsilon):
        time_ma = np.mean([self.episode_time[-100:]])
        T = np.sum(self.episode_time)

        template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
        template += 'Market: {:>6.1%} ({:>6.1%}) | '
        template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
        print(template.format(episode, format_time(total),
                              nav_ma_100 - 1, nav_ma_10 - 1,
                              market_nav_100 - 1, market_nav_10 - 1,
                              win_ratio, epsilon))

    def predict_single_day(self, trading_environment: Env, properties: SimulatorProperties,
                           ddqn_agent: DQNAgent):

        action = 0
        # Train bot using
        this_state = trading_environment.reset()
        for episode_step in range(properties.trading_days):
            # predict action (or get random)
            action = ddqn_agent.epsilon_greedy_policy(this_state.reshape(-1, ddqn_agent.state_dim))

        trading_environment.close()

        return action

    def run_step(self, trading_environment: Env, properties: SimulatorProperties,
                 ddqn_agent: DQNAgent):
        # Train bot using
        this_state = trading_environment.reset()
        for episode_step in range(properties.trading_days):
            # predict action (or get random)
            action = ddqn_agent.epsilon_greedy_policy(this_state.reshape(-1, ddqn_agent.state_dim))

            # check result
            next_state, reward, done, _ = trading_environment.step(action)

            # save step
            ddqn_agent.memorize_transition(this_state,
                                           action,
                                           reward,
                                           next_state,
                                           0.0 if done else 1.0)
            this_state = next_state
            print(f'Episode step: {episode_step}')

        # get DataFrame with seqence of actions, returns and nav values
        result: pd.DataFrame = trading_environment.env.simulator.result()
        trading_environment.close()

        Plotter.plot_nav(properties, result)

        results = pd.DataFrame(
            {
                'Date': result['dates'],
                'Agent': result['nav'],
                'Market': result['market_nav'],
                'Strategy Return': result['strategy_return'],
                'Start Position': result['start_position'],
                'Market Return': result['market_return'],
            }).set_index('Date')

        results.to_csv(f'{properties.get_path()}/data_frame_{properties.path}{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

        return

    def run_steps(self, trading_environment: Env, properties: SimulatorProperties,
                  ddqn_agent: DQNAgent):
        # Train bot using
        for episode in range(1, properties.max_episodes + 1):
            this_state = trading_environment.reset()
            for episode_step in range(properties.trading_days):

                # predict action (or get random)
                action = ddqn_agent.epsilon_greedy_policy(this_state.reshape(-1, ddqn_agent.state_dim))

                # check result
                next_state, reward, done, _ = trading_environment.step(action)

                # save step
                ddqn_agent.memorize_transition(this_state,
                                               action,
                                               reward,
                                               next_state,
                                               0.0 if done else 1.0)

                if ddqn_agent.train:
                    # train network
                    ddqn_agent.experience_replay()
                if done:
                    break
                this_state = next_state

            # get DataFrame with seqence of actions, returns and nav values
            result = trading_environment.env.simulator.result()

            # get results of last step
            final = result.iloc[-1]

            # apply return (net of cost) of last action to last starting nav
            nav = final.nav * (1 + final.strategy_return)
            self.navs.append(nav)

            # market nav
            market_nav = final.market_nav
            self.market_navs.append(market_nav)

            # track difference between agent an market NAV results
            diff = nav - market_nav
            self.diffs.append(diff)

            if episode % 10 == 0:
                self.track_results(episode,
                                   # show mov. average results for 100 (10) periods
                                   np.mean(self.navs[-100:]),
                                   np.mean(self.navs[-10:]),
                                   np.mean(self.market_navs[-100:]),
                                   np.mean(self.market_navs[-10:]),
                                   # share of agent wins, defined as higher ending nav
                                   np.sum([s > 0 for s in self.diffs[-100:]]) / min(len(self.diffs), 100),
                                   time() - self.start, ddqn_agent.epsilon)
            # if len(self.diffs) > 25 and all([r > 0 for r in self.diffs[-25:]]):
            #     print(result.tail())
            #     break

        ddqn_agent.save_network(properties.path)
        trading_environment.close()

        # Store Results
        results = pd.DataFrame({'Episode': list(range(1, episode + 1)),
                                'Agent': self.navs,
                                'Market': self.market_navs,
                                'Difference': self.diffs}).set_index('Episode')

        results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()

        return results

    def save_results(self, results, properties: SimulatorProperties):
        results.to_csv(f'{properties.get_path()}\\results_{properties.stock.ticker}.csv', index=False)

        Plotter.plot_performance(properties, results)
