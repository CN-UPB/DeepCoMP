import os
import logging

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
        self.env_class = config['env']
        self.env_name = config['env'].__name__
        self.env_config = config['env_config']
        self.episode_length = self.env_config['episode_length']

        # set agent
        supported_agents = ('ppo', 'random', 'fixed')
        assert agent_type in supported_agents, f"Agent {agent_type} not supported. Supported agents: {supported_agents}"
        self.agent_type = agent_type
        self.agent = None
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
            return ppo.PPOTrainer(config=config, env=self.env_class)
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

    def train(self, train_iter):
        def rllib_train(config, reporter):
            agent = self.get_agent(config)
            # collect number of episode steps and rewards over all training episodes for plotting later
            eps_steps = []
            eps_rewards = []
            for i in range(train_iter):
                results = agent.train()
                # collect and save episode steps and rewards in results
                # otherwise only the last 100 are saved by default
                # FIXME: somehow save or get all episode rewards for plotting the learning curve later
                #  this works, but in the end I only get a string, not a list of rewards. I can't seem to get this list out of the function
                #  https://github.com/ray-project/ray/issues/9104
                # eps_steps.extend(results['hist_stats']['episode_lengths'])
                # eps_rewards.extend(results['hist_stats']['episode_reward'])
                # results['eps_steps'] = eps_steps
                # results['eps_rewards'] = eps_rewards
                self.log.debug('Train iteration done', train_iter=i, results=results)
                reporter(**results)

            # the path is relative to tune's local_dir; try getting the path and loading the agent somehow
            # TODO: save inside experiment folder but also copy to training/trained_agent
            save_path = agent.save('../trained_agents/')
            self.log.info('Agent saved', path=save_path)

        # FIXME: need to implement a save function to use checkpiont in the end; or save inside the function
        analysis = ray.tune.run(rllib_train, config=self.config, local_dir=self.save_dir, checkpoint_at_end=False)
        self.log.info('Train results', train_results=analysis)
        return analysis

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

    def run(self, config, num_episodes=1, render=None, log_steps=False):
        """
        Run one simulation episode. Render situation at beginning of each time step. Return episode reward.
        :param config: RLlib config to create a new trainer/agent
        :param num_episodes: Number of episodes to run
        :param render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param log_steps: Whether or not to log infos about each step or just about each episode
        :return: Return list of episode rewards
        """
        assert (num_episodes == 1) or (render == None), "Turn off rendering when running for multiple episodes"
        eps_rewards = []
        # create and load agent
        self.agent = self.get_agent(config)
        self.agent.restore('../training/RLlibEnv/rllib_train/trained_agents/checkpoint_1/checkpoint-1')

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
                # in contrast to the logged step in the env, these obs, rewards, etc are processed (eg, clipped, normalized)
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

    def evaluate(self, eval_eps):
        """Evaluate the agent over specified number of episodes. Return avg & std episode reward and step reward"""
        mean_eps_reward, std_eps_reward = evaluate_policy(self.agent, self.env, n_eval_episodes=eval_eps)
        mean_step_reward = mean_eps_reward / self.episode_length
        self.log.info("Policy evaluation", mean_eps_reward=mean_eps_reward, std_eps_reward=std_eps_reward,
                      mean_step_reward=mean_step_reward)
        return mean_eps_reward, std_eps_reward, mean_step_reward
