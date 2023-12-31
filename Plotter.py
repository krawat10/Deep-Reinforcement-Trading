import json
import os
import re
from datetime import datetime
from time import mktime
from types import SimpleNamespace
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

from DataSource import DataSource
from Models.SimulationResult import SimulationResult
from Models.SimulatorProperties import SimulatorProperties


class Plotter:
    @staticmethod
    def plot_multiple_detailed_histograms(paths: List[str]):
        simulations = Plotter.get_simulations(paths)

        labels = []
        incorrect_shorts_counts = []
        correct_shorts_counts = []
        neutral_counts = []
        incorrect_longs_counts = []
        correct_longs_counts = []

        for simulation in simulations:
            labels.append(simulation.agent_to_pretty_string())

            # SHORT with negative reward
            incorrect_shorts = simulation.data.loc[
                lambda df: df['Start Position'] == -1].loc[
                lambda df: df['Strategy Return'] < 0].shape[0]

            # SHORT with positive reward
            correct_shorts = simulation.data.loc[
                lambda df: df['Start Position'] == -1].loc[
                lambda df: df['Strategy Return'] > 0].shape[0]

            # CASH positions
            neutral_count = simulation.data.loc[
                lambda df: df['Start Position'] == 0].shape[0]

            # LONG with negative reward
            incorrect_longs = simulation.data.loc[
                lambda df: df['Start Position'] == 1].loc[
                lambda df: df['Strategy Return'] < 0].shape[0]

            # LONG with positive reward
            correct_longs = simulation.data.loc[
                lambda df: df['Start Position'] == 1].loc[
                lambda df: df['Strategy Return'] > 0].shape[0]

            incorrect_shorts_counts.append(incorrect_shorts)
            correct_shorts_counts.append(correct_shorts)
            neutral_counts.append(neutral_count)
            incorrect_longs_counts.append(incorrect_longs)
            correct_longs_counts.append(correct_longs)

        x = np.arange(len(labels))  # the label locations
        width = 1.1  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 2 * (width / 6), incorrect_shorts_counts, width / 6, label='Incorrect Shorts', color='red')
        rects2 = ax.bar(x - width / 6, correct_shorts_counts, width / 6, label='Correct Shorts', color='green')
        rects3 = ax.bar(x, neutral_counts, width / 6, label='Neutral', color='gray')
        rects4 = ax.bar(x + width / 6, incorrect_longs_counts, width / 6, label='Incorrect Longs', color='tomato')
        rects5 = ax.bar(x + 2 * (width / 6), correct_longs_counts, width / 6, label='Correct Longs', color='lime')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Count')
        ax.set_title('Correct/Neutral/Incorrect choices')
        ax.set_xticks(x, labels, rotation=20)
        ax.legend(loc="lower right")

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3)
        ax.bar_label(rects5, padding=3)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiple_histograms(paths: List[str]):
        simulations = Plotter.get_simulations(paths)

        labels = []
        incorrect_counts = []
        correct_counts = []
        neutral_counts = []

        for simulation in simulations:
            labels.append(simulation.agent_to_pretty_string())

            # LONG/SHORT with negative reward
            incorrect_count = simulation.data.loc[
                lambda df: df['Start Position'] != 0].loc[
                lambda df: df['Strategy Return'] < 0].shape[0]

            # CASH positions
            neutral_count = simulation.data.loc[
                lambda df: df['Start Position'] == 0].shape[0]

            # LONG/SHORT with positive reward
            correct_count = simulation.data.loc[
                lambda df: df['Start Position'] != 0].loc[
                lambda df: df['Strategy Return'] > 0].shape[0]

            incorrect_counts.append(incorrect_count)
            neutral_counts.append(neutral_count)
            correct_counts.append(correct_count)

        x = np.arange(len(labels))  # the label locations
        width = 0.8  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 3, incorrect_counts, width / 3, label='Incorrect', color='red')
        rects2 = ax.bar(x, neutral_counts, width / 3, label='Neutral', color='gray')
        rects3 = ax.bar(x + width / 3, correct_counts, width / 3, label='Correct', color='green')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Count')
        ax.set_title('Correct/Neutral/Incorrect choices')
        ax.set_xticks(x, labels, rotation=20)
        ax.legend(loc="lower right")

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def get_simulations(paths):
        data_regex = re.compile('(data_frame_)(.*csv$)')
        config_regex = re.compile('(config.json)')
        simulations: List[SimulationResult] = []
        for path in paths:
            simulation = SimulationResult()
            simulation.path = path

            for root, dirs, files in os.walk(path):
                for file in files:
                    if data_regex.match(file):
                        simulation.data = pd.read_csv(f'{path}/{file}')
                    if config_regex.match(file):
                        f = open(f'{path}/{file}').read()
                        simulation.properties = json.loads(f, object_hook=lambda d: SimpleNamespace(**d))
            simulations.append(simulation)
        return simulations

    @staticmethod
    def plot_multiple_nav(paths: List[str]):
        simulation_list = Plotter.get_simulations(paths)
        time = pd.to_datetime(simulation_list[0].data['Date'])
        fig = plt.figure()

        plt.plot(time, simulation_list[0].data['Market'], linestyle='-.',
                 label=simulation_list[0].market_to_pretty_string())

        for simulation in simulation_list:
            plt.plot(time, simulation.data['Agent'], label=simulation.agent_to_pretty_string())

        plt.axhline(y=1.0)
        plt.xlabel('trading days')
        plt.ylabel('net asset value')
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.20))
        fig.autofmt_xdate()
        plt.show()
        return

    @staticmethod
    def plot_nav(properties: SimulatorProperties, result):
        # np.array convert timestamp to unix number and matplotlib does not recognize it.
        # Therefore we need to convert date to numeric format recognized by matplotlib
        x = result['dates'].apply(lambda date: mdates.epoch2num(mktime(date.timetuple())))
        y_market = result['market_nav']
        y_agent = result['nav']
        z = result['position']
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points_market = np.array([x, y_market]).T.reshape(-1, 1, 2)
        segments_market = np.concatenate([points_market[:-1], points_market[1:]], axis=1)
        points_agent = np.array([x, y_agent]).T.reshape(-1, 1, 2)
        segments_agent = np.concatenate([points_agent[:-1], points_agent[1:]], axis=1)
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        plt.xticks(rotation=30)
        # Create a continuous norm to map from data points to colors
        cmap = ListedColormap(['r', 'b', 'g'])
        norm = BoundaryNorm([-properties.stock.leverage, -0.5, 0.5, properties.stock.leverage], cmap.N)
        lc = LineCollection(segments_agent, cmap=cmap, norm=norm)
        lc.set_array(z)
        lc.set_linewidth(2)
        line = axs[0].add_collection(lc)
        fig.colorbar(line, ax=axs[0])
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[0].set_xlim(x.min(), x.max())
        axs[0].set_ylim(min(y_market.min(), y_agent.min()), max(y_market.max(), y_agent.max()))
        axs[0].set_title(f'Agent performance for stock {properties.stock.ticker}')
        axs[0].set_xlabel('trading days')
        axs[0].set_ylabel('net asset value')
        # Use a boundary norm instead
        cmap = ListedColormap(['r', 'b', 'g'])
        norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
        lc = LineCollection(segments_market, cmap=cmap, norm=norm)
        lc.set_array(z)
        lc.set_linewidth(2)
        line = axs[1].add_collection(lc)
        fig.colorbar(line, ax=axs[1])
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[1].set_xlim(x.min(), x.max())
        axs[1].set_ylim(min(y_market.min(), y_agent.min()), max(y_market.max(), y_agent.max()))
        axs[1].set_title(f'Market performance for stock {properties.stock.ticker}')
        axs[1].set_xlabel('trading days')
        axs[1].set_ylabel('net asset value')
        plt.savefig(f'{properties.get_path()}/nav_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()

    @staticmethod
    def plot_data_source(data_source: DataSource, path):
        start = data_source.ohlc.index.min()
        end = data_source.ohlc.index.max()
        plt.figure(figsize=(10, 10))
        plt.plot(data_source.ohlc.index, data_source.ohlc['close'])
        plt.xlabel("date")
        plt.ylabel("$ price")
        plt.title(f"{data_source.ticker} Price {start} - {end}")
        plt.legend()
        plt.savefig(f'{path}/chart.png')
        plt.show()

    @staticmethod
    def plot_performance(properties: SimulatorProperties, results):
        with sns.axes_style('white'):
            sns.distplot(results.Difference)
            sns.despine()
        # Evaluate Results
        results.info()
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
        df1 = (results[['Agent', 'Market']]
               .sub(1)
               .rolling(100)
               .mean())
        df1.plot(ax=axes[0],
                 title='Annual Returns (Moving Average)',
                 lw=1)
        df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
        df2.plot(ax=axes[1],
                 title='Agent Outperformance (%, Moving Average)')
        for ax in axes:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        axes[1].axhline(.5, ls='--', c='k', lw=1)
        sns.despine()
        fig.tight_layout()
        fig.savefig(f'{properties.path}/performance', dpi=300)
        fig.show()
