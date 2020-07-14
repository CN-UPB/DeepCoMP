"""Useful scripts for evaluation"""
import os
from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt

from drl_mobile.util.constants import TRAIN_DIR, TEST_DIR, PLOT_DIR


def read_training_progress(dir_name):
    """
    Read progress.csv from training an RLlib agent. Return a pandas data frame

    :param dir_name: Dir name of the training run, eg, 'PPO_MultiAgentMobileEnv_0_2020-07-13_15-42-542fjkk2px'
    :return: Tuple: Pandas data frame holding the original contents, Data frame only holding hist eps_rewards
    """
    # read file
    progress_file = os.path.join(TRAIN_DIR, f'{dir_name}/progress.csv')
    print(f"Reading file {progress_file}")
    df = pd.read_csv(progress_file)

    # convert hist eps rewards into proper lists (are read as strings): https://stackoverflow.com/a/32743458/2745116
    df['hist_stats/episode_reward'] = df['hist_stats/episode_reward'].apply(literal_eval)

    # create 2nd df for plotting with just episodes and eps_rewards
    # read all episode rewards from hist stats into long list
    eps_rewards = []
    for eps in df['hist_stats/episode_reward']:
        eps_rewards.extend([r for r in eps])
    episodes = [i+1 for i in range(len(eps_rewards))]
    df2 = pd.DataFrame(data={'episode': episodes, 'eps_reward': eps_rewards})

    return df, df2


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
def plot_eps_reward(dfs, labels, filename=None):
    """
    Plot the testing episode rewards over time

    :param dfs: List of data frames to plot and compare
    :param labels: List of labels corresponding to the data frames
    :param filename: Filename for the saved figure
    """
    assert len(dfs) <= len(labels), "Each data frame needs a label"

    # plot different data frames
    for i, df in enumerate(dfs):
        plt.plot(df['episode'], df['eps_reward'], label=labels[i])

    # axis and legend
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.legend()

    # saving
    if filename is not None:
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/{filename}')
    plt.show()


# for training results
def plot_mean_eps_reward(df):
    """Plot the mean episode reward per training iteration"""
    plt.plot(df['training_iteration'], df['episode_reward_mean'])
    plt.xlabel('Training Iteration')
    plt.ylabel('Mean Episode Reward')
    plt.show()


if __name__ == '__main__':
    df_ppo_org, df_ppo = read_training_progress('PPO_MultiAgentMobileEnv_0_2020-07-13_17-17-15pe9vn3ul')
    # plot_mean_eps_reward(df_ppo_org)
    df_rand = read_testing_results('RandomAgent_DatarateMobileEnv_2020-07-13_17-34-07.csv')
    df_rand2 = read_testing_results('RandomAgent_DatarateMobileEnv_2020-07-13_17-25-15.csv')

    plot_eps_reward([df_rand, df_rand2, df_ppo], ['rand1', 'rand2', 'ppo'], filename='eps_reward.pdf')
