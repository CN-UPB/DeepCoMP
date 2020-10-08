import os
import time
from datetime import datetime
from collections import defaultdict

import pandas as pd
import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import ray
import ray.tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.util.constants import SUPPORTED_ALGS, SUPPORTED_RENDER, RESULT_DIR, TRAIN_DIR, TEST_DIR, VIDEO_DIR
from drl_mobile.agent.dummy import RandomAgent, FixedAgent
from drl_mobile.agent.heuristics import GreedyBestSelection, GreedyAllSelection
from drl_mobile.util.logs import config_logging


class Simulation:
    """Simulation class for training and testing agents."""
    def __init__(self, config, agent_name, cli_args, debug=False):
        """
        Create a new simulation object to hold the agent and environment, train & test & visualize the agent + env.

        :param config: RLlib agent config
        :param agent_name: String identifying the agent. Supported: 'ppo', 'greedy-best', 'random', 'fixed'
        :param cli_args: Dict of CLI args
        :param debug: Whether or not to enable ray's local_mode for debugging
        """
        # config and env
        self.config = config
        self.env_class = config['env']
        self.env_name = config['env'].__name__
        self.env_config = config['env_config']
        self.episode_length = self.env_config['episode_length']
        # detect automatically if the env is a multi-agent env by checking all (not just immediate) ancestors
        self.multi_agent_env = MultiAgentEnv in self.env_class.__mro__
        # num workers for parallel execution of eval episodes
        self.num_workers = config['num_workers']
        self.cli_args = cli_args

        # agent
        assert agent_name in SUPPORTED_ALGS, f"Agent {agent_name} not supported. Supported agents: {SUPPORTED_ALGS}"
        self.agent_name = agent_name
        self.agent = None
        # only init ray if necessary --> lower overhead for dummy agents
        if self.agent_name == 'ppo':
            ray.init(local_mode=debug)

        # filename for saving is set when loading the agent
        self.result_filename = None

        self.log = structlog.get_logger()
        self.log.debug('Simulation init', env=self.env_name, eps_length=self.episode_length, agent=self.agent_name,
                       multi_agent=self.multi_agent_env, num_workers=self.num_workers)

    def train(self, stop_criteria, restore_path=None):
        """
        Train an RLlib agent using tune until any of the configured stopping criteria is met.

        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :param restore_path: Path to trained agent to continue training (if any)
            The agent's latest checkpoint is loaded automatically
            The trained agent needs to have the same settings and scenario for continuing training
            When continuing training, the number of training steps continues too, ie, is not reset to 0 after restoring
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # load latest checkpoint within the given agent's directory
        if restore_path is not None:
            restore_path = self.get_last_checkpoint_path(restore_path)

        analysis = ray.tune.run(PPOTrainer, config=self.config, local_dir=RESULT_DIR, stop=stop_criteria,
                                # checkpoint every 10 iterations and at the end; keep the best 10 checkpoints
                                checkpoint_at_end=True, checkpoint_freq=10, keep_checkpoints_num=10,
                                checkpoint_score_attr='episode_reward_mean', restore=restore_path)
        analysis.default_metric = 'episode_reward_mean'
        analysis.default_mode = 'max'

        # tune returns an ExperimentAnalysis that can be cast to a Pandas data frame
        # object https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis
        df = analysis.dataframe()
        checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial())
        self.log.info('Training done', timesteps_total=int(df['timesteps_total']),
                      episodes_total=int(df['episodes_total']), episode_reward_mean=float(df['episode_reward_mean']),
                      num_steps_sampled=int(df['info/num_steps_sampled']),
                      num_steps_trained=int(df['info/num_steps_trained']),
                      log_dir=analysis.get_best_logdir())

        # plot results
        # this only contains (and plots) the last 100 episodes --> not useful
        #  --> use tensorboard instead; or read and plot progress.csv
        # eps_results = df['hist_stats']
        # self.plot_learning_curve(eps_results['episode_lengths'], eps_results['episode_reward'])
        return checkpoint_path, analysis

    @staticmethod
    def get_last_checkpoint_path(rllib_dir):
        """Given an RLlib training dir, return the full path to the last checkpoint"""
        # check if rllib_dir is really already a pointer to a specific checkpoint; in that case, just return it
        if 'checkpoint' in rllib_dir and os.path.isfile(rllib_dir):
            return rllib_dir

        rllib_dir = os.path.abspath(rllib_dir)
        checkpoints = [f for f in os.listdir(rllib_dir) if f.startswith('checkpoint')]
        last_checkpoint_dir = os.path.join(rllib_dir, checkpoints[-1])
        # eg, retrieve '10' from '...PPO_MultiAgentMobileEnv_0_2020-07-14_17-28-33je5r1lov/checkpoint_10'
        last_checkpoint_no = last_checkpoint_dir.split('_')[-1]
        # construct full checkpoint path, eg, '...r1lov/checkpoint_10/checkpoint-10'
        last_checkpoint_path = os.path.join(last_checkpoint_dir, f'checkpoint-{last_checkpoint_no}')
        return last_checkpoint_path

    @staticmethod
    def get_best_checkpoint_path(rllib_dir):
        """Given an RLlib training dir, return the full path of the best checkpoint"""
        # check if rllib_dir is really already a pointer to a specific checkpoint; in that case, just return it
        if 'checkpoint' in rllib_dir and os.path.isfile(rllib_dir):
            return rllib_dir

        rllib_dir = os.path.abspath(rllib_dir)
        analysis = ray.tune.Analysis(rllib_dir)
        analysis.default_metric = 'episode_reward_mean'
        analysis.default_mode = 'max'
        checkpoint = analysis.get_best_checkpoint(analysis._get_trial_paths()[0])
        return os.path.abspath(checkpoint)

    def load_agent(self, rllib_dir=None, rand_seed=None, fixed_action=1, explore=False):
        """
        Load a trained RLlib agent from the specified rllib_path. Call this before testing a trained agent.

        :param rllib_dir: Path pointing to the agent's training dir (only used for RLlib agents)
        :param rand_seed: RNG seed used by the random agent (ignored by other agents)
        :param fixed_action: Fixed action performed by the fixed agent (ignored by the others)
        :param explore: Whether to keep exploration enabled. Set to False when testing an RLlib agent.
        True for continuing training.
        """
        checkpoint_path = None
        if self.agent_name == 'ppo':
            # turn off exploration for testing the loaded agent
            self.config['explore'] = explore
            self.agent = PPOTrainer(config=self.config, env=self.env_class)
            checkpoint_path = self.get_best_checkpoint_path(rllib_dir)
            self.log.info('Loading PPO agent', checkpoint=checkpoint_path)
            self.agent.restore(checkpoint_path)
        if self.agent_name == 'greedy-best':
            self.agent = GreedyBestSelection()
        if self.agent_name == 'greedy-all':
            self.agent = GreedyAllSelection()
        if self.agent_name == 'random':
            # instantiate the environment to get the action space
            env = self.env_class(self.env_config)
            self.agent = RandomAgent(env.action_space, seed=rand_seed)
        if self.agent_name == 'fixed':
            self.agent = FixedAgent(action=fixed_action, noop_interval=100)

        self.log.info('Agent loaded', agent=type(self.agent).__name__, rllib_dir=rllib_dir, checkpoint=checkpoint_path)

        # set a suitable filename for saving testing videos and results later
        self.set_result_filename()

    def set_result_filename(self):
        """Return a suitable filename (without file ending) in the format 'agent_env-class_env-size_num-ues_time'"""
        assert self.agent is not None, "Set the filename after loading the agent"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent_name = type(self.agent).__name__
        env_size = self.cli_args.env
        num_ues = self.cli_args.slow_ues + self.cli_args.fast_ues
        self.result_filename = f'{agent_name}_{self.env_name}_{env_size}_{num_ues}UEs_{timestamp}'

    def save_animation(self, fig, patches, mode):
        """
        Create and save matplotlib animation

        :param fig: Matplotlib figure
        :param patches: List of patches to draw for each step in the animation
        :param mode: How to save the animation. Options: 'video' (=html5) or 'gif' (requires ImageMagick)
        """
        render_modes = SUPPORTED_RENDER - {None}
        assert mode in render_modes, f"Render mode {mode} not in {render_modes}"
        anim = matplotlib.animation.ArtistAnimation(fig, patches, repeat=False)

        # save html5 video
        if mode == 'html' or mode == 'both':
            html = anim.to_html5_video()
            with open(f'{VIDEO_DIR}/{self.result_filename}.html', 'w') as f:
                f.write(html)
            self.log.info('Video saved', path=f'{VIDEO_DIR}/{self.result_filename}.html')

        # save gif; requires external dependency ImageMagick
        if mode == 'gif' or mode == 'both':
            try:
                anim.save(f'{VIDEO_DIR}/{self.result_filename}.gif', writer='imagemagick')
                self.log.info('Gif saved', path=f'{VIDEO_DIR}/{self.result_filename}.gif')
            except TypeError:
                self.log.error('ImageMagick needs to be installed for saving gifs.')

    def apply_action_single_agent(self, obs, env):
        """
        For the given observation and a trained/loaded agent, get and apply the next action. Only single-agent envs.

        :param dict obs: Dict of observations for all agents
        :param env: The environment to which to apply the actions to
        :returns: tuple (obs, r, done, info) WHERE
            obs is the next observation
            r is the immediate reward
            done is done['__all__'] indicating if all agents are done
        """
        assert not self.multi_agent_env, "Use apply_action_multi_agent for multi-agent envs"
        assert self.agent is not None, "Train or load an agent before running the simulation"
        action = self.agent.compute_action(obs)
        next_obs, reward, done, info = env.step(action)
        self.log.debug("Step", t=info['time'], obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        return next_obs, reward, done, info

    def apply_action_multi_agent(self, obs, env):
        """
        Same as apply_action_single_agent, but for multi-agent envs. For each agent, unpack obs & choose action,
        before applying it to the env.

        :param dict obs: Dict of observations for all agents
        :param env: The environment to which to apply the actions to
        :returns: tuple (obs, r, done, info) WHERE
            obs is the next observation
            r is the summed up immediate reward for all agents
            done is done['__all__'] indicating if all agents are done
        """
        assert self.multi_agent_env, "Use apply_action_single_agent for single-agent envs"
        assert self.agent is not None, "Train or load an agent before running the simulation"
        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = self.config['multiagent']['policy_mapping_fn'](agent_id)
            action[agent_id] = self.agent.compute_action(agent_obs, policy_id=policy_id)
        next_obs, reward, done, info = env.step(action)
        # info is currently the same for all agents; just get the info from the last agent
        info = info[agent_id]
        self.log.debug("Step", t=info['time'], obs=obs, action=action, reward=reward, next_obs=next_obs,
                       done=done['__all__'])
        return next_obs, sum(reward.values()), done['__all__'], info

    def run_episode(self, env, render=None, log_dict=None):
        """
        Run a single episode on the given environment. Append episode reward and exec time to list and return.

        :param env: Instance of the environment to use (each joblib iteration will still use its own instance)
        :param render: Whether/How to render the episode
        :param log_dict: Dict with logging levels to set
        :return: Tuple of eps_duration (scalar), step rewards (list), metrics per step (list of dicts)
        """
        # list of rewards and metrics (which are a dict) for each time step
        rewards = []
        scalar_metrics = []

        # no need to instantiate new env since each joblib iteration has its own copy
        # that's why we need to set the logging level again for each iteration
        config_logging()
        if log_dict is not None:
            env.set_log_level(log_dict)

        eps_start = time.time()
        if render is not None:
            fig = plt.figure(figsize=env.map.figsize)
            # equal aspect ratio to avoid distortions
            plt.gca().set_aspect('equal')

        # run until episode ends
        patches = []
        t = 0
        done = False
        obs = env.reset()
        # for continuous problems, stop evaluation after fixed eps length
        while not done and t < self.episode_length:
            if render is not None:
                patches.append(env.render())
                if render == 'plot':
                    plt.show()

            # get and apply action
            if self.multi_agent_env:
                obs, reward, done, info = self.apply_action_multi_agent(obs, env)
            else:
                obs, reward, done, info = self.apply_action_single_agent(obs, env)
            t = info['time']

            # save reward and metrics
            rewards.append(reward)
            scalar_metrics.append(info['scalar_metrics'])

        # create the animation
        if render is not None:
            fig.tight_layout()
            self.save_animation(fig, patches, render)

        # episode time in seconds (to measure simulation efficiency)
        eps_duration = time.time() - eps_start

        self.log.debug('Episode complete', eps_duration=eps_duration, avg_step_reward=np.mean(rewards),
                       metrics=list(scalar_metrics[0].keys()))
        # TODO: if returning the full rewards and metrics per step consumes too much memory, aggregate them here
        #  using scipy.stats.describe https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html
        return eps_duration, rewards, scalar_metrics

    @staticmethod
    def summarize_results(eps_duration, rewards, scalar_metrics):
        """
        Summarize given results into single result dict containing everything that should be logged and written to file.

        :param eps_duration: List of episode durations (in s)
        :param rewards: List of lists with rewards per step per episode
        :param scalar_metrics: List of lists, containing a dict of metric --> value for each episode for each time step
        :returns: Dict of result name --> whatever should be logged and saved (eg, mean, std, etc)
        """
        results = defaultdict(list)
        num_episodes = len(eps_duration)
        # get metric names from first metric dict (first episode, first step); it's the same for all steps and eps
        metric_names = list(scalar_metrics[0][0].keys())

        # iterate over all episodes and aggregate the results per episode
        for e in range(num_episodes):
            # add episode, eps_duration and rewards
            results['episode'].append(e + 1)
            results['eps_duration_mean'].append(eps_duration[e])
            results['eps_duration_std'].append(eps_duration[e])
            results['step_reward_mean'].append(np.mean(rewards[e]))
            results['step_reward_std'].append(np.std(rewards[e]))

            # calc mean and std per metric and episode
            for metric in metric_names:
                metric_values = [scalar_metrics[e][t][metric] for t in range(len(scalar_metrics[e]))]
                results[f'{metric}_mean'].append(np.mean(metric_values))
                results[f'{metric}_std'].append(np.std(metric_values))

        # convert defaultdict to normal dict
        return dict(results)

    def write_results(self, results):
        """Write experiment results to CSV file. Include all relevant info."""
        result_file = f'{TEST_DIR}/{self.result_filename}.csv'
        self.log.info("Writing results", file=result_file)

        # distinguish multi-agent RL with separate NNs rather than a shared NN for all agents
        agent_str = self.cli_args.agent
        if agent_str == 'multi' and self.cli_args.separate_agent_nns:
            agent_str = 'multi-sep-nns'

        # input/configuration data to track to what the results belong to
        data = {
            'alg': self.cli_args.alg,
            'agent': agent_str,
            'env': self.env_name,
            'env_size': self.cli_args.env,
            'eps_length': self.episode_length,
            'num_bs': len(self.env_config['bs_list']),
            'sharing_model': self.cli_args.sharing,
            'num_ue_slow': self.cli_args.slow_ues,
            'num_ue_fast': self.cli_args.fast_ues
        }
        # training data for PPO
        if self.agent_name == 'ppo':
            data.update({
                'train_steps': self.cli_args.train_steps,
                'train-iter': self.cli_args.train_iter,
                'target_reward': self.cli_args.target_reward
            })

        # add actual results and save to file
        data.update(results)
        df = pd.DataFrame(data=data)
        df.to_csv(result_file)

    def run(self, num_episodes=1, render=None, log_dict=None, write_results=False):
        """
        Run one or more simulation episodes. Render situation at beginning of each time step. Return episode rewards.

        :param int num_episodes: Number of episodes to run
        :param str render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param dict log_dict: Dict of logger names --> logging level used to configure logging in the environment
        :param bool write_results: Whether or not to write experiment results to file
        :return list: Return list of lists with step rewards for all episodes
        """
        assert self.agent is not None, "Train or load an agent before running the simulation"
        assert (num_episodes == 1) or (render is None), "Turn off rendering when running for multiple episodes"
        if self.agent_name == 'ppo' and self.num_workers > 1:
            self.log.warning('PPO testing and evaluation cannot be parallelized. Continuing with 1 worker.')
            self.num_workers = 1

        # instantiate env and set logging level
        env = self.env_class(self.env_config)
        if log_dict is not None:
            env.set_log_level(log_dict)

        # simulate episodes in parallel; show progress with tqdm if running for more than one episode
        self.log.info('Starting evaluation', num_episodes=num_episodes, num_workers=self.num_workers,
                      slow_ues=self.cli_args.slow_ues, fast_ues=self.cli_args.fast_ues)
        # run episodes in parallel using joblib
        zipped_results = Parallel(n_jobs=self.num_workers)(
            delayed(self.run_episode)(env, render, log_dict)
            for _ in tqdm(range(num_episodes), disable=(num_episodes == 1))
        )
        # results consisting of list of tuples with (eps_duration, rewards, scalar_metrics) for each episode
        # unzip to separate lists with entries for each episode (rewards and metrics are lists of lists; for each step)
        eps_duration, rewards, scalar_metrics = map(list, zip(*zipped_results))

        # summarize results
        results = self.summarize_results(eps_duration, rewards, scalar_metrics)
        self.log.info('Summarized results', results=results)
        self.log.info("Simulation complete", num_episodes=num_episodes, eps_length=self.episode_length,
                      step_reward_mean=np.mean(results['step_reward_mean']),
                      step_reward_std=np.std(results['step_reward_std']),
                      avg_eps_reward=self.episode_length * np.mean(results['step_reward_mean']))

        # write results to file
        if write_results:
            self.write_results(results)

        return rewards
