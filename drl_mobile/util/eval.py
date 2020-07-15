"""Useful scripts for evaluation"""
import os
from ast import literal_eval
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from drl_mobile.util.constants import TRAIN_DIR, TEST_DIR, EVAL_DIR, PLOT_DIR


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
    for i, eps in enumerate(df['hist_stats/episode_reward']):
        # attention: the lists overlap for different iterations! only read the results from this iteration
        # FIXME: they still do
        num_eps = df['episodes_this_iter'][i]
        eps_rewards.extend([r for r in eps[:-num_eps]])
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
def plot_eps_reward(dfs, labels, roll_mean_window=1, filename=None):
    """
    Plot the testing episode rewards over time

    :param dfs: List of data frames to plot and compare
    :param labels: List of labels corresponding to the data frames
    :param filename: Filename for the saved figure
    """
    assert len(dfs) <= len(labels), "Each data frame needs a label"

    # plot different data frames
    for i, df in enumerate(dfs):
        plt.plot(df['episode'], df['eps_reward'].rolling(window=roll_mean_window).mean(), label=labels[i])

    # axis and legend
    plt.xlim([0, 800])
    plt.xlabel('Episode')
    plt.ylabel(f'Mean Episode Reward (Rolling Window of {roll_mean_window})')
    plt.legend()

    # saving
    if filename is not None:
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/{filename}')
    plt.show()


# for training results
def plot_ppo_mean_eps_reward(df):
    """Plot the mean episode reward per training iteration"""
    plt.plot(df['episodes_total'], df['episode_reward_mean'], label='PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Episode Reward')
    eps_per_iter = np.mean(df['episodes_this_iter'])
    return int(eps_per_iter)


# summarizing results from different runs
def summarize_results(dir=EVAL_DIR):
    """Read and summarize all results in a directory. Return a df."""
    config_cols = ['alg', 'agent', 'num_ue_slow', 'num_ue_fast', 'eps_length', 'env_size']
    result_cols = ['eps_reward', 'eps_dr', 'eps_util', 'eps_unsucc_conn', 'eps_lost_conn']
    files = [f for f in os.listdir(dir) if f.endswith('.csv')]
    data = defaultdict(list)

    # read all files and save relevant data
    for f in files:
        df = pd.read_csv(f'{dir}/{f}')
        # copy config
        for conf in config_cols:
            data[conf].append(df[conf][0])
        # summarize num ues
        data['num_ue'].append(df['num_ue_slow'][0] + df['num_ue_fast'][0])
        # summarize eps reward and other stats
        data['num_eps'].append(len(df['eps_reward']))
        # calc mean and std for all results
        for res in result_cols:
            data[f'{res}_mean'].append(np.mean(df[res]))
            data[f'{res}_std'].append(np.std(df[res]))

    # create and return combined df
    return pd.DataFrame(data=data)


def concat_results(dir=EVAL_DIR):
    """Read results and concat df for plotting with seaborn"""
    dfs = []
    for f in os.listdir(dir):
        dfs.append(pd.read_csv(f'{dir}/{f}'))
    return pd.concat(dfs)


def plot_increasing_ues(df, metric, filename=None):
    """Plot results for increasing num. UEs. Takes summarized df as input."""
    for alg in df['alg'].unique():
        df_alg = df[df['alg'] == alg]

        # for PPO, distinguish between centralized and multi-agent
        if alg == 'ppo':
            for agent in df_alg['agent'].unique():
                df_ppo = df_alg[df_alg['agent'] == agent]
                plt.errorbar(df_ppo['num_ue'], df_ppo[f'{metric}_mean'], yerr=df_ppo[f'{metric}_std'], capsize=5,
                             label=f'{alg}_{agent}')

        else:
            plt.errorbar(df_alg['num_ue'], df_alg[f'{metric}_mean'], yerr=df_alg[f'{metric}_std'], capsize=5, label=alg)

    # axes and legend
    plt.xlabel("Num. UEs")
    plt.ylabel(metric)

    # remove error bars from legend: https://stackoverflow.com/a/15551976/2745116
    # get handles
    # handles, labels = plt.get_legend_handles_labels()
    # # remove the errorbars
    # handles = [h[0] for h in handles]
    # # use them in the legend
    # plt.legend(handles, labels, numpoints=1)

    plt.legend()

    # saving
    if filename is not None:
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/{filename}')
    plt.show()


if __name__ == '__main__':
    # df_ppo_org, df_ppo = read_training_progress('PPO_MultiAgentMobileEnv_0_2020-07-14_09-34-32asp5wtp5')
    # df_greedy_best = read_testing_results('GreedyBestSelection_MultiAgentMobileEnv_2020-07-14_10-52-16.csv')
    # df_greedy_all = read_testing_results('GreedyAllSelection_MultiAgentMobileEnv_2020-07-14_10-56-44.csv')
    # df_rand = read_testing_results('RandomAgent_CentralMultiUserEnv_2020-07-14_11-07-26.csv')
    #
    # dfs = [df_greedy_best, df_greedy_all, df_rand]
    # labels = ['Greedy-Best', 'Greedy-All', 'Random']
    #
    # eps_per_iter = plot_ppo_mean_eps_reward(df_ppo_org)
    # plot_eps_reward(dfs, labels, roll_mean_window=eps_per_iter, filename='eps_reward.pdf')

    df = summarize_results(dir=f'{EVAL_DIR}/comparison/incr_ues')
    # df = concat_results()
    plot_increasing_ues(df, metric='eps_reward', filename='reward_incr_ues.pdf')
    plot_increasing_ues(df, metric='eps_dr', filename='dr_incr_ues.pdf')
    plot_increasing_ues(df, metric='eps_util', filename='utility_incr_ues.pdf')
    plot_increasing_ues(df, metric='eps_unsucc_conn', filename='unsucc_conn_incr_ues.pdf')
    plot_increasing_ues(df, metric='eps_lost_conn', filename='lost_conn_incr_ues.pdf')
