import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
from stable_baselines import results_plotter
from stable_baselines.common.evaluation import evaluate_policy


log = structlog.get_logger()


class Simulation:
    """Simulation class"""
    def __init__(self, env, agent):
        self.env = env
        self.env_name = type(env).__name__
        self.agent = agent

    def train(self, train_steps, save_dir, plot=False):
        """Train agent for specified training steps"""
        log.info('Start training', train_steps=train_steps)
        self.agent.learn(train_steps)
        self.agent.save(f'{save_dir}/ppo2_{train_steps}')
        if plot:
            results_plotter.plot_results([save_dir], train_steps, results_plotter.X_TIMESTEPS,
                                         f'Learning Curve for {self.env_name}')
            plt.savefig(f'{save_dir}/ppo2_{train_steps}.png')
            plt.show()

    def run(self, render=None, save_dir=None):
        """Run one simulation episode. Render situation at beginning of each time step. Return episode reward."""
        if render is not None:
            # square figure and equal aspect ratio to avoid distortions
            fig = plt.figure(figsize=(5, 5))
            plt.gca().set_aspect('equal')

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
            episode_reward += reward
        if render is not None:
            patches.append(self.env.render())
            # it's either plt.show or saving the video; both doesn't work (would prob. work with plt.draw)
            if render == 'plot':
                plt.show()
            if render == 'video':
                assert save_dir is not None, 'You must specify save_dir if rendering video'
                anim = matplotlib.animation.ArtistAnimation(fig, patches, repeat=False)
                html = anim.to_html5_video()
                with open(f'{save_dir}/replay.html', 'w') as f:
                    f.write(html)
                log.info('Video saved', path=f'{save_dir}/replay.html')

        log.info('Simulation complete', episode_reward=episode_reward)
        return episode_reward

    def evaluate(self, eval_eps):
        """Evaluate the agent over specified number of episodes. Return avg & std episode reward and step reward"""
        mean_eps_reward, std_eps_reward = evaluate_policy(self.agent, self.env, n_eval_episodes=eval_eps)
        mean_step_reward = mean_eps_reward / self.env.episode_length
        log.info("Policy evaluation", mean_eps_reward=mean_eps_reward, std_eps_reward=std_eps_reward,
                 mean_step_reward=mean_step_reward)
        return mean_eps_reward, std_eps_reward, mean_step_reward
