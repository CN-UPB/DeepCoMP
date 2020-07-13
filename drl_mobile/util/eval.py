"""Useful scripts for evaluation"""
import os
import pathlib
from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt


def read_training_progress(dir_name):
    """
    Read progress.csv from training an RLlib agent. Return a pandas data frame

    :param dir_name: Dir name of the training run, eg, 'PPO_MultiAgentMobileEnv_0_2020-07-13_15-42-542fjkk2px'
    :return: Pandas data frame holding the contents
    """
    # construct path that works independent from where the script is run
    this_dir = pathlib.Path(__file__).parent.absolute()
    project_root = this_dir.parent.parent.absolute()
    progress_file = os.path.join(project_root, f'training/PPO/{dir_name}/progress.csv')

    # read progress.csv
    print(f"Reading file {progress_file}")
    df = pd.read_csv(progress_file)
    # convert hist eps rewards into proper lists (are read as strings): https://stackoverflow.com/a/32743458/2745116
    df['hist_stats/episode_reward'] = df['hist_stats/episode_reward'].apply(literal_eval)

    return df


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
    df = read_training_progress('PPO_MultiAgentMobileEnv_0_2020-07-13_15-42-542fjkk2px')
    plot_mean_eps_reward(df)
    plot_full_eps_reward(df)
