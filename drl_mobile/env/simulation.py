import os

import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
# from stable_baselines import results_plotter
# from stable_baselines.common.evaluation import evaluate_policy
import seaborn as sns
import numpy as np
import ray.tune
import ray.rllib.agents.ppo as ppo


class Simulation:
    """Simulation class"""
    def __init__(self, config, agent_type, normalize):
        # save the config
        self.config = config
        self.env = config['env']
        self.env_name = config['env'].__name__
        self.env_config = config['env_config']
        self.episode_length = self.env_config['episode_length']

        # set agent
        supported_agents = ('ppo', 'random', 'fixed')
        assert agent_type in supported_agents, f"Agent {agent_type} not supported. Supported agents: {supported_agents}"
        self.agent_type = agent_type
        self.normalize = normalize

        # dir for saving logs, plots, replay video
        self.save_dir = f'../training/{self.env_name}'
        os.makedirs(self.save_dir, exist_ok=True)

        self.log = structlog.get_logger()

    def get_agent(self, config):
        """
        Return an agent object of the configured type
        :param config: Config to pass to the agent
        :return: Created agent object
        """
        if self.agent_type == 'ppo':
            return ppo.PPOTrainer(config=config, env=self.env)
        else:
            raise NotImplementedError(f"Agent {self.agent_type} not yet implemented")

    def train_sb(self, train_steps, save_dir, plot=False):
        """Train stable_baselines agent for specified training steps"""
        self.log.info('Start training', train_steps=train_steps)
        self.agent.learn(train_steps)
        self.agent.save(f'{save_dir}/ppo2_{train_steps}')
        if self.normalize:
            self.env.save(f'{save_dir}/vec_norm.pkl')
        if plot:
            results_plotter.plot_results([save_dir], train_steps, results_plotter.X_TIMESTEPS,
                                         f'Learning Curve for {self.env_name}')
            plt.savefig(f'{save_dir}/ppo2_{train_steps}.png')
            plt.show()

    def plot_training_results(self, eps_steps, eps_rewards, plot_eps=True):
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

    def train_rllib(self, train_iter, save_dir, plot=False):
        """Train RLlib agent"""
        self.log.info('Start training', total_train_iter=train_iter)
        # # TODO: configure training length; plot progress; save
        results = {}
        for i in range(train_iter):
            results = self.agent.train()
            self.log.debug('Train iteration done', train_iter=i, results=results)
        eps_results = results['hist_stats']
        # FIXME: this only contains the last 100 episodes --> not useful; instead properly configer the log dir and then plot progress.csv
        self.plot_training_results(eps_results['episode_lengths'], eps_results['episode_reward'])

    def train_rllib2(self, train_iter):
        def custom_tune_loop(config, reporter):
            agent = self.get_agent(config)
            for i in range(train_iter):
                results = agent.train()
                self.log.debug('Train iteration done', train_iter=i, results=results)
                reporter(**results)

        final_results = ray.tune.run(custom_tune_loop, config=self.config, local_dir=self.save_dir)

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

    def run(self, render=None, save_dir=None):
        """
        Run one simulation episode. Render situation at beginning of each time step. Return episode reward.
        :param render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param save_dir: Where to save rendered HTML5 video or gif (directory)
        """
        if render is not None:
            # square figure and equal aspect ratio to avoid distortions
            fig = plt.figure(figsize=(5, 5))
            plt.gca().set_aspect('equal')
            fig.tight_layout()

        # run until episode ends
        patches = []
        episode_reward = 0
        done = False
        obs = self.env.reset()
        while not done:
            if render is not None:
                patches.append(self.env.render())
                if render == 'plot':
                    plt.show()
            # deterministic=True is important: https://github.com/hill-a/stable-baselines/issues/832
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            # in contrast to the logged step in the env, these obs, rewards, etc are processed (eg, clipped, normalized)
            self.log.info("Step", action=action, reward=reward, next_obs=obs, done=done)
            episode_reward += reward
        # VecEnv is directly reset when episode ends, so we cannot show the end of the episode after the final step
        # https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html

        # create the animation
        if render == 'video' or render == 'gif':
            self.save_animation(fig, patches, render, save_dir)

        self.log.info('Simulation complete', episode_reward=episode_reward)
        return episode_reward

    def evaluate(self, eval_eps):
        """Evaluate the agent over specified number of episodes. Return avg & std episode reward and step reward"""
        mean_eps_reward, std_eps_reward = evaluate_policy(self.agent, self.env, n_eval_episodes=eval_eps)
        mean_step_reward = mean_eps_reward / self.episode_length
        self.log.info("Policy evaluation", mean_eps_reward=mean_eps_reward, std_eps_reward=std_eps_reward,
                      mean_step_reward=mean_step_reward)
        return mean_eps_reward, std_eps_reward, mean_step_reward
