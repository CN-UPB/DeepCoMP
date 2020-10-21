"""Useful scripts for evaluation"""
import os
import glob
from ast import literal_eval
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from drl_mobile.util.constants import TRAIN_DIR, TEST_DIR, PLOT_DIR, RESULT_DIR, PROJECT_ROOT


# assuming the b5g-results repo is checked out next to this repo
EVAL_DIR = os.path.join(PROJECT_ROOT, os.pardir, 'b5g-results', 'drl_mobile')
# EVAL_DIR = os.path.join(B5G_RESULT_DIR, '2020-10-21_eval-draft')


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


def get_result_files(dir, prefix='', suffix='.csv', skip_folder='/old/'):
    """Read all files in the directory recursivley and return a list of all files matching the prefix and suffix"""
    result_files = []
    for f in glob.iglob(dir + '**/**', recursive=True):
        # skip files in given skip_folder, only consider results in /test/ subdir
        if skip_folder in f or 'test' not in f:
            continue
        # only select files (not dirs) that are in a '/test/' subdir (to filter out PPO's progress.csv)
        if os.path.isfile(f) and f.startswith(prefix) and f.endswith(suffix):
            result_files.append(f)
    return result_files


# summarizing results from different runs
def summarize_results(dir=EVAL_DIR, read_hist_data=False):
    """Read and summarize all results in a directory. Return a df."""
    config_cols = ['alg', 'agent', 'num_ue_slow', 'num_ue_fast', 'eps_length', 'env_size']
    result_cols = ['avg_utility']
    # files = [f for f in os.listdir(dir) if f.endswith('.csv')]
    files = get_result_files(dir)
    data = defaultdict(list)

    # read all files and save relevant data
    for f in files:
        df = pd.read_csv(f)
        # copy config
        for conf in config_cols:
            data[conf].append(df[conf][0])
        # summarize num ues
        num_ue = df['num_ue_slow'][0] + df['num_ue_fast'][0]
        data['num_ue'].append(num_ue)
        # summarize eps reward and other stats
        # data['num_eps'].append(len(df['eps_reward']))
        # calc mean and std for all results
        for res in result_cols:
            data[f'{res}_mean'].append(df[f'{res}_mean'])
            data[f'{res}_std'].append(df[f'{res}_std'])
            data[f'{res}_sum'].append(df[f'{res}_mean'] * num_ue)

        # read individual data rates and utilities for histogram plotting
        if read_hist_data:
            # convert into proper lists: https://stackoverflow.com/a/32743458/2745116
            data['dr_list'] = [df['dr_list'].apply(literal_eval)]
            data['utility_list'] = [df['utility_list'].apply(literal_eval)]

    # Calculate and add reliability to the df, defined as `num_no_conn / (num_ue * eps_length)`, ie,
    # percent of steps with any connection (no matter the dr) averaged over all UEs
    # data['reliability'] = np.array(data['num_no_conn_mean']) / (data['num_ue'][0] * data['eps_length'][0])

    # create and return combined df
    return pd.DataFrame(data=data)


def concat_results(dir=EVAL_DIR):
    """Read results and concat df for plotting with seaborn"""
    dfs = []
    for f in os.listdir(dir):
        dfs.append(pd.read_csv(f'{dir}/{f}'))
    return pd.concat(dfs)


def plot_increasing_ues(df, metric, plot_sum=True, filename=None):
    """Plot results for increasing num. UEs. Takes summarized df as input."""
    for alg in df['alg'].unique():
        if alg == 'random':
            continue
        df_alg = df[df['alg'] == alg]
        for agent in df_alg['agent'].unique():
            df_agent = df_alg[df_alg['agent'] == agent]

            # plot 'metric' summed up over all UEs
            if plot_sum:
                plt.plot(df_agent['num_ue'], df_agent[f'{metric}_sum'], label=f'{alg}_{agent}', marker=True)
            # or plot the metric averaged over all UEs
            else:
                plt.errorbar(df_agent['num_ue'], df_agent[f'{metric}_mean'], yerr=df_agent[f'{metric}_std'], capsize=5,
                             label=f'{alg}_{agent}', marker=True)

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


def plot_histogram(df, metric, cdf=False):
    """Plot a histogram of the given metric"""
    assert metric in {'dr_list', 'utility_list'}, "Currently only 'dr_list' and 'utility_list' are supported"
    # expand df where each row contains a list to a df with separate rows for each unit
    plt.hist(df[metric].explode(), density=True, cumulative=cdf)
    plt.xlabel(metric)
    plt.ylabel(f'Density (Cumulative: {cdf})')
    # FIXME: there's something wrong with the data rates
    # plt.xlim(0, 10000)
    plt.show()


if __name__ == '__main__':
    # df = summarize_results(dir=f'{RESULT_DIR}')
    df = summarize_results(dir=f'{EVAL_DIR}/2020-10-21_eval-draft')
    # plot_increasing_ues(df, metric='eps_reward', filename='reward_incr_ues.pdf')
    # plot_increasing_ues(df, metric='eps_dr', filename='dr_incr_ues.pdf')
    plot_increasing_ues(df, metric='avg_utility', filename='utility_incr_ues.pdf')
    # plot_increasing_ues(df, metric='eps_unsucc_conn', filename='unsucc_conn_incr_ues.pdf')
    # plot_increasing_ues(df, metric='eps_lost_conn', filename='lost_conn_incr_ues.pdf')
