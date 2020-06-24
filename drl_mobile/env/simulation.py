import os
import logging

import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
import seaborn as sns
import numpy as np
import ray.tune
import ray.rllib.agents.ppo as ppo


class Simulation:
    """Simulation class"""
    def __init__(self, config, agent_type):
        # config and env
        self.config = config
        self.env_class = config['env']
        self.env_name = config['env'].__name__
        self.env_config = config['env_config']
        self.episode_length = self.env_config['episode_length']

        # agent
        # TODO: test random and fixed agent
        supported_agents = ('ppo', 'random', 'fixed')
        assert agent_type in supported_agents, f"Agent {agent_type} not supported. Supported agents: {supported_agents}"
        self.agent_type = agent_type
        self.agent = None

        # save dir
        self.save_dir = f'../training'
        os.makedirs(self.save_dir, exist_ok=True)

        self.log = structlog.get_logger()

    def plot_learning_curve(self, eps_steps, eps_rewards, plot_eps=True):
        """
        Plot episode rewards over time
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
        :return: Return a Pandas data frame with the training results
        """
        analysis = ray.tune.run(ppo.PPOTrainer, config=self.config, local_dir=self.save_dir, stop=stop_criteria,
                                checkpoint_at_end=True)
        # tune returns an ExperimentAnalysis that can be cast to a Pandas data frame
        # object https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis
        df = analysis.dataframe()
        self.log.info('Training done', timesteps_total=int(df['timesteps_total']),
                      episodes_total=int(df['episodes_total']), episode_reward_mean=float(df['episode_reward_mean']),
                      num_steps_sampled=int(df['info/num_steps_sampled']),
                      num_steps_trained=int(df['info/num_steps_trained']))

        # plot results
        # TODO: this only contains (and plots) the last 100 episodes --> not useful
        #  --> use tensorboard instead; or read and plot progress.csv
        # eps_results = df['hist_stats']
        # self.plot_learning_curve(eps_results['episode_lengths'], eps_results['episode_reward'])
        return df

    def load_agent(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this when testing without training up front.
        :param path: Path pointing to the agent's saved checkpoint
        """
        self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
        self.agent.restore(path)
        self.log.info('Agent loaded', agent=self.agent, path=path)

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
            env.set_log_level('drl_mobile.env.simulation', logging.DEBUG)
        else:
            env.set_log_level('drl_mobile.env.simulation', logging.INFO)

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
                # deterministic=True is important: https://github.com/hill-a/stable-baselines/issues/832
                # action, _states = self.agent.predict(obs, deterministic=True)
                action = self.agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                # in contrast to the logged step in the env, these obs, rewards, etc may be further processed
                # (eg, clipped, normalized)
                self.log.debug("Step", action=action, reward=reward, next_obs=obs, done=done)
                episode_reward += reward
            # VecEnv is directly reset when episode ends, so we cannot show the end of the episode after the final step
            # https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html

            # create the animation
            if render == 'video' or render == 'gif':
                self.save_animation(fig, patches, render, self.save_dir)

            eps_rewards.append(episode_reward)
            self.log.info('Episode complete', episode_reward=episode_reward)

        # summarize episode rewards
        mean_eps_reward = np.mean(eps_rewards)
        mean_step_reward = mean_eps_reward / self.episode_length
        self.log.info("Simulation complete", mean_eps_reward=mean_eps_reward, std_eps_reward=np.std(eps_rewards),
                      mean_step_reward=mean_step_reward)
        return eps_rewards
