import os

import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
import seaborn as sns
import numpy as np
import ray
import ray.tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv

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
        # detect automatically if the env is a multi-agent env by checking all (not just immediate) ancestors
        self.multi_agent_env = MultiAgentEnv in self.env_class.__mro__

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
        self.log.debug('Simulation init', env=self.env_name, eps_length=self.episode_length, agent=self.agent_name,
                       multi_agent=self.multi_agent_env)

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

    def save_animation(self, fig, patches, mode):
        """
        Create and save matplotlib animation

        :param fig: Matplotlib figure
        :param patches: List of patches to draw for each step in the animation
        :param mode: How to save the animation. Options: 'video' (=html5) or 'gif' (requires ImageMagick)
        """
        render_modes = ('html', 'gif', 'both')
        assert mode in render_modes, f"Render mode {mode} not in {render_modes}"
        anim = matplotlib.animation.ArtistAnimation(fig, patches, repeat=False)

        # save html5 video
        if mode == 'video' or mode == 'both':
            html = anim.to_html5_video()
            with open(f'{self.save_dir}/{self.env_name}.html', 'w') as f:
                f.write(html)
            self.log.info('Video saved', path=f'{self.save_dir}/{self.env_name}.html')

        # save gif; requires external dependency ImageMagick
        if mode == 'gif' or mode == 'both':
            try:
                anim.save(f'{self.save_dir}/{self.env_name}.gif', writer='imagemagick')
                self.log.info('Gif saved', path=f'{self.save_dir}/{self.env_name}.gif')
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

    def run(self, num_episodes=1, render=None, log_dict=None):
        """
        Run one or more simulation episodes. Render situation at beginning of each time step. Return episode rewards.

        :param int num_episodes: Number of episodes to run
        :param str render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param dict log_dict: Dict of logger names --> logging level used to configure logging in the environment
        :return list: Return list of episode rewards
        """
        assert self.agent is not None, "Train or load an agent before running the simulation"
        assert (num_episodes == 1) or (render == None), "Turn off rendering when running for multiple episodes"
        eps_rewards = []

        # instantiate env and set logging level
        env = self.env_class(self.env_config)
        if log_dict is not None:
            env.set_log_level(log_dict)

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

                # get and apply action, increment episode reward
                if self.multi_agent_env:
                    obs, reward, done = self.apply_action_multi_agent(obs, env)
                else:
                    obs, reward, done = self.apply_action_single_agent(obs, env)
                episode_reward += reward

            # create the animation
            if render is not None:
                self.save_animation(fig, patches, render)

            eps_rewards.append(episode_reward)
            self.log.info('Episode complete', episode_reward=episode_reward)

        # summarize episode rewards
        mean_eps_reward = np.mean(eps_rewards)
        mean_step_reward = mean_eps_reward / self.episode_length
        self.log.info("Simulation complete", mean_eps_reward=mean_eps_reward, std_eps_reward=np.std(eps_rewards),
                      mean_step_reward=mean_step_reward, num_episodes=num_episodes)
        return eps_rewards
