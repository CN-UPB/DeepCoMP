import time
from datetime import datetime

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

from drl_mobile.util.constants import SUPPORTED_ALGS, SUPPORTED_RENDER, RESULT_DIR, TRAIN_DIR, EVAL_DIR, VIDEO_DIR
from drl_mobile.agent.dummy import RandomAgent, FixedAgent
from drl_mobile.agent.heuristics import GreedyBestSelection, GreedyAllSelection


class Simulation:
    """Simulation class for training and testing agents."""
    def __init__(self, config, agent_name, debug=False):
        """
        Create a new simulation object to hold the agent and environment, train & test & visualize the agent + env.

        :param config: RLlib agent config
        :param agent_name: String identifying the agent. Supported: 'ppo', 'greedy-best', 'random', 'fixed'
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

    def train(self, stop_criteria):
        """
        Train an RLlib agent using tune until any of the configured stopping criteria is met.

        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        analysis = ray.tune.run(PPOTrainer, config=self.config, local_dir=RESULT_DIR, stop=stop_criteria,
                                checkpoint_at_end=True)
        # tune returns an ExperimentAnalysis that can be cast to a Pandas data frame
        # object https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis
        df = analysis.dataframe()
        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean'),
                                                           metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        self.log.info('Training done', timesteps_total=int(df['timesteps_total']),
                      episodes_total=int(df['episodes_total']), episode_reward_mean=float(df['episode_reward_mean']),
                      num_steps_sampled=int(df['info/num_steps_sampled']),
                      num_steps_trained=int(df['info/num_steps_trained']),
                      log_dir=analysis.get_best_logdir(metric='episode_reward_mean'))

        # plot results
        # this only contains (and plots) the last 100 episodes --> not useful
        #  --> use tensorboard instead; or read and plot progress.csv
        # eps_results = df['hist_stats']
        # self.plot_learning_curve(eps_results['episode_lengths'], eps_results['episode_reward'])
        return checkpoint_path, analysis

    def load_agent(self, rllib_path=None, rand_seed=None, fixed_action=1):
        """
        Load a trained RLlib agent from the specified rllib_path. Call this before testing a trained agent.

        :param rllib_path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        :param rand_seed: RNG seed used by the random agent (ignored by other agents)
        :param fixed_action: Fixed action performed by the fixed agent (ignored by the others)
        """
        if self.agent_name == 'ppo':
            self.agent = PPOTrainer(config=self.config, env=self.env_class)
            self.agent.restore(rllib_path)
        if self.agent_name == 'greedy-best':
            self.agent = GreedyBestSelection()
        if self.agent_name == 'greedy-all':
            self.agent = GreedyAllSelection()
        if self.agent_name == 'random':
            # instantiate the environment to get the action space
            env = self.env_class(self.env_config)
            self.agent = RandomAgent(env.action_space, seed=rand_seed)
        if self.agent_name == 'fixed':
            self.agent = FixedAgent(action=fixed_action, noop_interval=4)

        self.log.info('Agent loaded', agent=type(self.agent).__name__, rllib_path=rllib_path)

        # set a suitable filename for saving testing videos and results later
        self.set_result_filename()

    def set_result_filename(self):
        """Return a suitable filename (without file ending) in the format 'agent_env_timestamp'"""
        assert self.agent is not None, "Set the filename after loading the agent"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent_name = type(self.agent).__name__
        self.result_filename = f'{agent_name}_{self.env_name}_{timestamp}'

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
        :returns: tuple (obs, r, done) WHERE
            obs is the next observation
            r is the immediate reward
            done is done['__all__'] indicating if all agents are done
        """
        assert not self.multi_agent_env, "Use apply_action_multi_agent for multi-agent envs"
        assert self.agent is not None, "Train or load an agent before running the simulation"
        action = self.agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        self.log.debug("Step", t=info['time'], action=action, reward=reward, next_obs=obs, done=done)
        return obs, reward, done

    def apply_action_multi_agent(self, obs, env):
        """
        Same as apply_action_single_agent, but for multi-agent envs. For each agent, unpack obs & choose action,
        before applying it to the env.

        :param dict obs: Dict of observations for all agents
        :param env: The environment to which to apply the actions to
        :returns: tuple (obs, r, done) WHERE
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
        obs, reward, done, info = env.step(action)
        # time is the same for all agents; just retrieve it from the last one
        time = info[agent_id]['time']
        self.log.debug("Step", t=time, action=action, reward=reward, next_obs=obs, done=done['__all__'])
        return obs, sum(reward.values()), done['__all__']

    def run_episode(self, env, render=None, log_dict=None):
        """
        Run a single episode on the given environment. Append episode reward and exec time to list and return.

        :param env: Instance of the environment to use (each joblib iteration will still use its own instance)
        :param render: Whether/How to render the episode
        :param log_dict: Dict with logging levels to set
        :return: Tuple(episode reward, execution time)
        """
        # no need to instantiate new env since each joblib iteration has its own copy
        # that's why we need to set the logging level again for each iteration
        if log_dict is not None:
            env.set_log_level(log_dict)

        eps_start = time.time()
        if render is not None:
            fig = plt.figure(figsize=(9, 6))
            # equal aspect ratio to avoid distortions
            plt.gca().set_aspect('equal')
            fig.tight_layout()

        # run until episode ends
        patches = []
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            if render is not None:
                patches.append(env.render())
                if render == 'plot':
                    plt.show()

            # get and apply action, increment episode reward
            if self.multi_agent_env:
                obs, reward, done = self.apply_action_multi_agent(obs, env)
            else:
                obs, reward, done = self.apply_action_single_agent(obs, env)
            episode_reward += reward

        # create the animation
        if render is not None:
            self.save_animation(fig, patches, render)

        # episode time in seconds (to measure simulation efficiency)
        eps_time = time.time() - eps_start
        self.log.debug('Episode complete', episode_reward=episode_reward, episode_time=eps_time)
        return episode_reward, eps_time

    def run(self, num_episodes=1, render=None, log_dict=None, write_results=False):
        """
        Run one or more simulation episodes. Render situation at beginning of each time step. Return episode rewards.

        :param int num_episodes: Number of episodes to run
        :param str render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param dict log_dict: Dict of logger names --> logging level used to configure logging in the environment
        :param bool write_results: Whether or not to write experiment results to file
        :return list: Return list of episode rewards
        """
        assert self.agent is not None, "Train or load an agent before running the simulation"
        assert (num_episodes == 1) or (render is None), "Turn off rendering when running for multiple episodes"

        # instantiate env and set logging level
        env = self.env_class(self.env_config)
        if log_dict is not None:
            env.set_log_level(log_dict)

        # simulate episodes in parallel; show progress with tqdm if running for more than one episode
        self.log.info('Starting evaluation', num_episodes=num_episodes, num_workers=self.num_workers)
        # run episodes sequentially
        # for _ in tqdm(range(num_episodes), disable=(num_episodes == 1)):
        #     self.run_episode(env, render)
        # run episodes in parallel using joblib
        zipped_results = Parallel(n_jobs=self.num_workers)(
            delayed(self.run_episode)(env, render, log_dict)
            for _ in tqdm(range(num_episodes), disable=(num_episodes == 1))
        )
        # unzip results, ie, convert list of tuples to two separate lists
        eps_rewards, eps_times = map(list, zip(*zipped_results))

        # summarize episode rewards
        mean_eps_reward = np.mean(eps_rewards)
        mean_step_reward = mean_eps_reward / self.episode_length
        self.log.info("Simulation complete", mean_eps_reward=mean_eps_reward, std_eps_reward=np.std(eps_rewards),
                      mean_step_reward=mean_step_reward, num_episodes=num_episodes,
                      mean_eps_time=np.mean(eps_times), std_eps_time=np.std(eps_times))

        # write results to file
        if write_results:
            result_file = f'{EVAL_DIR}/{self.result_filename}.csv'
            self.log.info("Writing results", file=result_file)
            # prepare and write result data
            data = {
                'episode': [i+1 for i in range(len(eps_rewards))],
                'eps_reward': eps_rewards
            }
            df = pd.DataFrame(data=data)
            df.to_csv(result_file)

        return eps_rewards
