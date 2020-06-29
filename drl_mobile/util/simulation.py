import os
import logging

import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
import seaborn as sns
import numpy as np
import ray
import ray.tune
from ray.rllib.agents.ppo import PPOTrainer

from drl_mobile.agent.dummy import RandomAgent, FixedAgent


class Simulation:
    """Simulation class for training and testing agents."""
    def __init__(self, config, agent_name, debug=False):
        """
        Create a new simulation object to hold the agent and environment, train & test & visualize the agent + env.
        :param config: RLlib agent config
        :param agent_name: String identifying the agent. Supported: 'ppo', 'random', 'fixed'
        :param debug: Whether or not to enable ray's local_mode for debugging
        """
        # config and env
        self.config = config
        self.env_class = config['env']
        self.env_name = config['env'].__name__
        self.env_config = config['env_config']
        self.episode_length = self.env_config['episode_length']

        # agent
        supported_agents = ('ppo', 'random', 'fixed')
        assert agent_name in supported_agents, f"Agent {agent_name} not supported. Supported agents: {supported_agents}"
        self.agent_name = agent_name
        self.agent = None
        # only init ray if necessary --> lower overhead for dummy agents
        if self.agent_name == 'ppo':
            ray.init(local_mode=debug)

        # save dir
        self.save_dir = f'../training'
        os.makedirs(self.save_dir, exist_ok=True)

        self.log = structlog.get_logger()

    def plot_learning_curve(self, eps_steps, eps_rewards, plot_eps=True):
        """
        Plot episode rewards over time. Currently not used.
        :param eps_steps: List of time steps per episode (should always be the same here)
        :param eps_rewards: List of reward per episode
        :param plot_eps: If true, plot episodes on the xlabel instead of steps
        """
        x = []
        if plot_eps:
            # just create a list of training episodes (not time steps)
            x = [i+1 for i in range(len(eps_rewards))]
        else:
            # sum up episode time steps
            for i, t in enumerate(eps_steps):
                if i == 0:
                    x.append(t)
                else:
                    x.append(t + x[i-1])

        # plot
        sns.regplot(x, eps_rewards, x_estimator=np.mean)
        plt.title(f'Learning Curve for {self.env_name}')
        if plot_eps:
            plt.xlabel('Training episodes')
        else:
            plt.xlabel('Training steps')
        plt.ylabel('Episode reward')
        train_steps = sum(eps_steps)
        plt.savefig(f'{self.save_dir}/rllib_ppo_{train_steps}.png')
        plt.show()

    def train(self, stop_criteria):
        """
        Train an RLlib agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        analysis = ray.tune.run(PPOTrainer, config=self.config, local_dir=self.save_dir, stop=stop_criteria,
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
        if self.agent_name == 'random':
            # instantiate the environment to get the action space
            env = self.env_class(self.env_config)
            self.agent = RandomAgent(env.action_space, seed=rand_seed)
        if self.agent_name == 'fixed':
            self.agent = FixedAgent(action=fixed_action, noop_interval=4)

        self.log.info('Agent loaded', agent=type(self.agent).__name__, rllib_path=rllib_path)

    def save_animation(self, fig, patches, mode, save_dir):
        """
        Create and save matplotlib animation
        :param fig: Matplotlib figure
        :param patches: List of patches to draw for each step in the animation
        :param mode: How to save the animation. Options: 'video' (=html5) or 'gif' (requires ImageMagick)
        :param save_dir: In which directory to save the animation
        """
        assert mode == 'video' or mode == 'gif', "Mode for saving animation must be 'video' or 'gif'"
        assert save_dir is not None, 'You must specify a save_dir for saving video/gif'
        anim = matplotlib.animation.ArtistAnimation(fig, patches, repeat=False)

        # save html5 video
        if mode == 'video':
            html = anim.to_html5_video()
            with open(f'{save_dir}/replay.html', 'w') as f:
                f.write(html)
            self.log.info('Video saved', path=f'{save_dir}/replay.html')

        # save gif; requires external dependency ImageMagick
        if mode == 'gif':
            try:
                anim.save(f'{save_dir}/replay.gif', writer='imagemagick')
                self.log.info('Gif saved', path=f'{save_dir}/replay.gif')
            except TypeError:
                self.log.error('ImageMagick needs to be installed for saving gifs.')

    def run(self, num_episodes=1, render=None, log_steps=False):
        """
        Run one or more simulation episodes. Render situation at beginning of each time step. Return episode rewards.
        :param config: RLlib config to create a new trainer/agent
        :param num_episodes: Number of episodes to run
        :param render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param log_steps: Whether or not to log infos about each step or just about each episode
        :return: Return list of episode rewards
        """
        assert self.agent is not None, "Train or load an agent before running the simulation"
        assert (num_episodes == 1) or (render == None), "Turn off rendering when running for multiple episodes"
        eps_rewards = []

        # instantiate env and set logging level
        env = self.env_class(self.env_config)
        if log_steps:
            env.set_log_level('drl_mobile.util.simulation', logging.DEBUG)
        else:
            env.set_log_level('drl_mobile.util.simulation', logging.INFO)

        # simulate for given number of episodes
        for _ in range(num_episodes):
            if render is not None:
                # square figure and equal aspect ratio to avoid distortions
                fig = plt.figure(figsize=(5, 5))
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
                # TODO: automatically set policy_id for multi-agent
                # FIXME: fix testing with multi-agent. probably sth wrong with action/obs shape
                action = self.agent.compute_action(obs, policy_id='ue')
                obs, reward, done, info = env.step(action)
                self.log.debug("Step", t=info['time'], action=action, reward=reward, next_obs=obs, done=done)
                episode_reward += reward

            # create the animation
            if render == 'video' or render == 'gif':
                self.save_animation(fig, patches, render, self.save_dir)

            eps_rewards.append(episode_reward)
            self.log.info('Episode complete', episode_reward=episode_reward)

        # summarize episode rewards
        mean_eps_reward = np.mean(eps_rewards)
        mean_step_reward = mean_eps_reward / self.episode_length
        self.log.info("Simulation complete", mean_eps_reward=mean_eps_reward, std_eps_reward=np.std(eps_rewards),
                      mean_step_reward=mean_step_reward, num_episodes=num_episodes)
        return eps_rewards
