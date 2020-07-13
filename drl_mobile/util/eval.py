"""Useful scripts for evaluation"""
import os
from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt

from drl_mobile.util.constants import PROJECT_ROOT, TRAIN_DIR, TEST_DIR


def read_training_progress(dir_name):
    """
    Read progress.csv from training an RLlib agent. Return a pandas data frame

    :param dir_name: Dir name of the training run, eg, 'PPO_MultiAgentMobileEnv_0_2020-07-13_15-42-542fjkk2px'
    :return: Pandas data frame holding the contents
    """
    progress_file = os.path.join(TRAIN_DIR, f'{dir_name}/progress.csv')
    print(f"Reading file {progress_file}")
    df = pd.read_csv(progress_file)
    # convert hist eps rewards into proper lists (are read as strings): https://stackoverflow.com/a/32743458/2745116
    df['hist_stats/episode_reward'] = df['hist_stats/episode_reward'].apply(literal_eval)

    return df


def read_testing_results(filename):
    """
    Read simulation testing results from csv file.

    :param filename: Filename, eg, 'RandomAgent_DatarateMobileEnv_2020-07-13_17-34-07.csv'
    :return: Data frame containing the results
    """
    result_file = os.path.join(TEST_DIR, filename)
    df = pd.read_csv(result_file)
    return df


# for testing results
def plot_eps_reward(df):
    """Plot the testing episode rewards over time"""
    plt.plot(df['episode'], df['eps_reward'])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.show()


# for training results
def plot_mean_eps_reward(df):
    """Plot the mean episode reward per training iteration"""
    plt.plot(df['training_iteration'], df['episode_reward_mean'])
    plt.xlabel('Training Iteration')
    plt.ylabel('Mean Episode Reward')
    plt.show()


def plot_full_eps_reward(df):
    """The all episode rewards (from the hist stats)"""
    # read all episode rewards from hist stats into long list
    eps_rewards = []
    for eps in df['hist_stats/episode_reward']:
        eps_rewards.extend([r for r in eps])
    # plot
    episodes = [i+1 for i in range(len(eps_rewards))]
    plt.plot(episodes, eps_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.show()


if __name__ == '__main__':
    # df = read_training_progress('PPO_MultiAgentMobileEnv_0_2020-07-13_17-17-15pe9vn3ul')
    # plot_mean_eps_reward(df)
    # plot_full_eps_reward(df)

    df = read_testing_results('RandomAgent_DatarateMobileEnv_2020-07-13_17-34-07.csv')
    plot_eps_reward(df)
