import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
# from stable_baselines import results_plotter
# from stable_baselines.common.evaluation import evaluate_policy


class Simulation:
    """Simulation class"""
    def __init__(self, env, agent, normalize):
        # we work with a dummy vec env with just 1 env; eg, for normalization
        # self.env = env
        # original_env = self.env.envs[0].env
        # self.env_name = type(original_env).__name__

        # and RLlib envs differently
        self.env_name = agent.config['env']
        self.env_config = agent.config['env_config']
        self.episode_length = self.env_config['episode_length']
        self.agent = agent
        self.normalize = normalize
        self.log = structlog.get_logger()

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

    def train_rllib(self, train_iter, save_dir, plot=False):
        """Train RLlib agent"""
        self.log.info('Start training', total_train_iter=train_iter)
        # TODO: configure training length; plot progress; save
        for i in range(train_iter):
            results = self.agent.train()
            self.log.debug('Train iteration done', train_iter=i, results=results)
        return results


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
