import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from random import sample

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import tensorflow as tf
from keras import Sequential, models
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

from AI.Models.DQNAgentType import DQNAgentType
from AI.Models.DecisionMakingPolicy import DecisionMakingPolicy
from AI.DropoutNoisy import DropoutNoisy


class DQNAgent:
    online_network: Sequential
    target_network: Sequential

    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size,
                 decision_making_policy,
                 dqn_type,
                 saved_network_dir=None):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.decision_making_policy: DecisionMakingPolicy = decision_making_policy
        self.dqn_type: DQNAgentType = dqn_type

        if saved_network_dir:
            self.online_network = self.load_model(f'{saved_network_dir}/online_network')
            self.target_network = self.load_model(f'{saved_network_dir}/target_network')
            if Path(f'{saved_network_dir}/experience.pkl').exists():
                self.experience = pickle.load(open(f'{saved_network_dir}/experience.pkl', 'rb'))
            self.print_weights()
        else:
            self.online_network = self.build_model()
            self.target_network = self.build_model(trainable=False)
            self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def load_model(self, dir: str):
        model = models.load_model(dir)
        return model

    def build_model(self, trainable=True) -> Sequential:
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        if self.decision_making_policy == DecisionMakingPolicy.NOISE_NETWORK_POLICY:
            layers.append(DropoutNoisy(units=self.architecture[-1],
                                       rate=0.1,
                                       std=0.1,
                                       activation='relu',
                                       kernel_regularizer=l2(self.l2_reg),
                                       name=f'DropoutNoisy_0',
                                       trainable=trainable))
            layers.append(Dense(units=self.num_actions,
                                trainable=trainable,
                                name='Output'))
        else:
            layers.append(Dropout(.1))
            layers.append(Dense(units=self.num_actions,
                                trainable=trainable,
                                name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1

        if self.decision_making_policy == DecisionMakingPolicy.EPSILON_GREEDY_POLICY:
            # Set random action (based on epsilon)
            if np.random.rand() <= self.epsilon:
                return np.random.choice(self.num_actions)
            else:
                q = self.online_network.predict(state)
                return np.argmax(q, axis=1).squeeze()

        # Select always random decisions
        if self.decision_making_policy == DecisionMakingPolicy.RANDOM_POLICY:
            return np.random.choice(self.num_actions)

        if self.decision_making_policy == DecisionMakingPolicy.NOISE_NETWORK_POLICY:
            q = self.online_network.predict(state)
            return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, current_state, action, reward, next_state, not_done):
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            # if last step of episode (last trading day)
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        # save state, action, reward, next_state
        self.experience.append((current_state, action, reward, next_state, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))  # to array with batch_size rows
        states, actions, rewards, next_states, not_done = minibatch  # to list with batch_size elements

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        # Evaluate choice using target network
        if self.dqn_type == DQNAgentType.DDQN:
            next_q_values_target = self.target_network.predict_on_batch(next_states)
        else:
            next_q_values_target = self.online_network.predict_on_batch(next_states)

        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[(self.idx, actions)] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()

    def save_network(self, path: str):
        self.target_network.save(f'{path}/target_network', True)
        self.online_network.save(f'{path}/online_network', True)
        pickle.dump(self.experience, open(f'{path}/experience.pkl', 'wb'))

    def print_weights(self):
        layers = self.online_network.layers

        output = '('
        for neuron in layers[0].weights:
            output += f' {neuron} '
        output += ')'
        print(output)
