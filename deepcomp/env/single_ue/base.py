"""Base mobile environment. Implemented and extended by sublcasses."""
import copy
import random
import logging

import gym
import gym.spaces
import structlog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patheffects as pe

from deepcomp.util.logs import config_logging
from deepcomp.env.util.utility import step_utility, log_utility
from deepcomp.env.entities.user import User
from deepcomp.env.util.movement import RandomWaypoint


class MobileEnv(gym.Env):
    """
    Base environment class with moving UEs and stationary BS on a map. RLlib and OpenAI Gym-compatible.
    No observation or action space implemented. This needs to be done in subclasses.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        """
        Create a new environment object with an OpenAI Gym interface. Required fields in the env_config:

        * episode_length: Total number of simulation time steps in one episode
        * map: Map object representing the playground
        * bs_list: List of base station objects in the environment
        * ue_list: List of UE objects in the environment
        * seed: Seed for the RNG; for reproducibility. May be None.

        :param env_config: Dict containing all configuration options for the environment. Required by RLlib.
        """
        super().__init__()
        self.time = 0
        self.episode_length = env_config['episode_length']
        self.map = env_config['map']
        self.bs_list = env_config['bs_list']
        self.ue_list = env_config['ue_list']
        # keep a copy of the original list, so it can easily be restored in reset()
        # shallow copy. If I just assign, it points to the same object. If I deepcopy, it copies and generates new UEs
        self.original_ue_list = copy.copy(env_config['ue_list'])
        self.new_ue_interval = env_config['new_ue_interval']
        # seed the environment
        self.env_seed = env_config['seed']
        self.seed(env_config['seed'])
        self.rand_episodes = env_config['rand_episodes']
        self.log_metrics = env_config['log_metrics']

        # current observation
        self.obs = None
        # observation and action space are defined in the subclass --> different variants
        self.observation_space = None
        self.action_space = None

        # configure logging inside env to ensure it works in ray/rllib. https://github.com/ray-project/ray/issues/9030
        config_logging()
        self.log = structlog.get_logger()
        self.log.info('Env init', env_config=env_config)

        # call after initializing everything else (needs settings, log)
        self.max_ues = self.get_max_num_ue()

    @property
    def num_bs(self):
        return len(self.bs_list)

    @property
    def num_ue(self):
        return len(self.ue_list)

    @property
    def total_dr(self):
        return sum([ue.curr_dr for ue in self.ue_list])

    @property
    def total_utility(self):
        return sum([ue.utility for ue in self.ue_list])

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            # seed RNG of map (used to generate new UEs' arrival points)
            self.map.seed(seed)
            # seed the RNG of all UEs (which themselves seed their movement)
            offset = 0
            for ue in self.ue_list:
                # add an offset to each UE's seed to avoid that all UEs have the same "random" pos and movement
                # use a fixed, not random offset here; otherwise it's again different in test & train
                offset += 100
                ue.seed(seed + offset)

    def set_log_level(self, log_dict):
        """
        Set a logging levels for a set of given logger. Needs to happen here, inside the env, for RLlib workers to work.
        :param dict log_dict: Dict with logger name --> logging level (eg, logging.INFO)
        """
        for logger_name, level in log_dict.items():
            logging.getLogger(logger_name).setLevel(level)

    def get_ue_obs(self, ue):
        """Return the an observation of the current world for a given UE"""
        raise NotImplementedError('Implement in subclass')

    def calc_reward(self, ue, penalty):
        """
        Calculate and return reward for specific UE: The UE's utility (based on its data rate) + penalty
        """
        # clip utility to -20, 20 to avoid -inf for 0 dr and cap at 100 dr
        clip_util = np.clip(ue.utility, -20, 20)

        # add penalty and clip again to stay in range -20, 20
        return np.clip(clip_util + penalty, -20, 20) / 20

    def reset(self):
        """Reset environment by resetting time and all UEs (pos & movement) and their connections"""
        if not self.rand_episodes:
            # seed again before every reset --> always run same episodes. agent behavior may still differ
            self.seed(self.env_seed)

        self.time = 0
        # reset UE list to avoid growing it infinitely
        old_list = self.ue_list
        self.ue_list = copy.copy(self.original_ue_list)
        # delete UEs that are no longer on the list
        for ue in set(old_list) - set(self.ue_list):
            del ue
        del old_list

        # reset all UEs and BS
        for ue in self.ue_list:
            ue.reset()
        for bs in self.bs_list:
            bs.reset()
        return self.get_obs()

    def get_max_num_ue(self):
        """Get the maximum number of UEs within an episode based on the new UE interval"""
        max_ues = self.num_ue
        if self.new_ue_interval is not None:
            # calculate the max number of UEs if one new UE is added at a given interval
            # eps_length - 1 because time is increased before checking done and t=eps_length is never reached
            max_ues = self.num_ue + int((self.episode_length - 1) / self.new_ue_interval)
            self.log.info('Num. UEs varies over time.', new_ue_interval=self.new_ue_interval, num_ues=self.num_ue,
                          max_ues=max_ues, episode_length=self.episode_length)
        return max_ues

    def get_ue_actions(self, action):
        """
        Retrieve the action per UE from the RL agent's action and return in in form of a dict.
        Does not yet apply actions to env.

        Here, in the single agent case, just get the action of a single active UE.
        Overwritten in other child envs, eg, for multi or central agent.

        :param action: Action that depends on the agent type (single, central, multi)
        :return: Dict that consistently (indep. of agent type) maps UE (object) --> action
        """
        assert self.action_space.contains(action), f"Action {action} does not fit action space {self.action_space}"
        # default to noop action
        action_dict = {ue: 0 for ue in self.ue_list}

        # select active UE (to update in this step) using round robin and get corresponding action
        ue = self.ue_list[self.time % self.num_ue]
        action_dict[ue] = action
        return action_dict

    def apply_ue_actions(self, action_dict):
        """
        Given a dict of UE actions, apply them to the environment and return a dict of penalties.
        This function remains unchanged and is simply inherited by all envs.

        Current penalty: -3 for any connect/disconnect action (for overhead)

        :param: Dict of UE --> action
        :return: Dict UE --> penalty
        """
        penalties = {ue: 0 for ue in self.ue_list}

        for ue, action in action_dict.items():
            # apply action: try to connect to BS; or: 0 = no op
            if action > 0:
                bs = self.bs_list[action-1]
                connected = ue.connect_to_bs(bs, disconnect=True, return_connected=True)

                # penalty -5 for connecting to a BS that's already in use by other UEs
                # if connected and bs.num_conn_ues >= 2:
                #     penalties[ue] = -5
                # # and +5 for disconnecting from a BS that's used by others
                # if not connected and bs.num_conn_ues >= 1:
                #     penalties[ue] = +5

                # penalty -3 for any connect/disconnect (whether successful or not)
                # penalties[ue] = -3

        # # add a penalty for concurrent connections (overhead for joint transmission), ie, for any 2+ connections
        # # tunable penalty weight representing the cost of concurrent connections
        # weight = 0
        # connections = len(ue.bs_dr)
        # if connections > 1:
        #     penalty -= weight * (connections - 1)

        return penalties

    def test_ue_actions(self, action_dict):
        """
        Test a given set of UE actions by applying them to a copy of the environment and return the rewards.

        :param action_dict: Actions to apply. Dict of UE --> action
        :return: Rewards for each UE when applying the action
        """
        # for proportional-fair sharing, also keep track of EWMA dr as it affects the allocated resources
        original_ewma_drs = {ue: ue.ewma_dr for ue in self.ue_list}
        self.apply_ue_actions(action_dict)
        # update ewma; need to update drs first for EWMA calculation to be correct
        self.update_ue_drs_rewards(penalties=None, update_only=True)
        for ue in self.ue_list:
            ue.update_ewma_dr()
        rewards = self.update_ue_drs_rewards(penalties=None)

        # to revert the action, apply it again: toggles connection again between same UE-BS
        self.apply_ue_actions(action_dict)
        # revert the ewma
        for ue in self.ue_list:
            ue.ewma_dr = original_ewma_drs[ue]
        self.update_ue_drs_rewards(penalties=None, update_only=True)

        # simpler + can be parallelized: just create a copy of the env, apply actions, get reward, delete copy
        # no, doesn't work: It's MUUUUUUUUUUUUUCH slower (probably due to all the deep copies)
        # test_env = copy.deepcopy(self)
        # test_env.apply_ue_actions(action_dict)
        # rewards = test_env.update_ue_drs_rewards(penalties=None)
        # del test_env
        return rewards

    def update_ue_drs_rewards(self, penalties, update_only=False):
        """
        Update cached data rates of all UE-BS connections.
        Calculate and return corresponding rewards based on given penalties.

        :param penalties: Dict of penalties for all UEs. Used for calculating rewards.
        :return: Dict of rewards: UE --> reward (incl. penalty)
        """
        rewards = dict()
        for ue in self.ue_list:
            ue.update_curr_dr()
            # calc and return reward if needed
            if update_only:
                # only update drs and return reward 0
                rewards[ue] = 0
            else:
                if penalties is None or ue not in penalties.keys():
                    rewards[ue] = self.calc_reward(ue, penalty=0)
                else:
                    rewards[ue] = self.calc_reward(ue, penalties[ue])
        return rewards

    def move_ues(self):
        """
        Move all UEs and return dict of penalties corresponding to number of lost connections.

        :return: Dict with num lost connections: UE --> num. lost connections
        """
        lost_conn = dict()
        for ue in self.ue_list:
            num_lost_conn = ue.move()
            # add penalty of -1 for each lost connection through movement (rather than actively disconnected)
            lost_conn[ue] = num_lost_conn
        return lost_conn

    def get_obs(self):
        """
        Return the current observation. Called to get the next observation after a step.
        Here, the obs for the next UE. Overwritten by env vars as needed.

        :returns: Next observation
        """
        next_ue = self.ue_list[self.time % self.num_ue]
        return self.get_ue_obs(next_ue)

    def step_reward(self, rewards):
        """
        Return the overall reward for the step (called at the end of a step). Overwritten by variants.

        :param rewards: Dict of avg rewards per UE (before and after movement)
        :returns: Reward for the step (depends on the env variant; here just for one UE)
        """
        # here: get reward for UE that applied the action (at time - 1)
        ue = self.ue_list[(self.time-1) % self.num_ue]
        return rewards[ue]

    def done(self):
        """
        Return whether the episode is done.

        :return: Whether the current episode is done or not
        """
        # return self.time >= self.episode_length
        # always return done=None to indicate that there is no natural episode ending
        # by default, the env is reset by RLlib and simulator when horizon=eps_length is reached
        # for continuous training, it the env is never reset during training
        return None

    def info(self):
        """
        Return info dict that's returned after a step. Includes info about current time step and metrics of choice.
        The metrics can be adjusted as required, but need to be structured into scalar_metrics and vector_metrics
        to be handled automatically.
        """
        if not self.log_metrics:
            return {'time': self.time}

        info_dict = {
            'time': self.time,
            # scalar metrics are metrics that consist of a single number/value per step
            'scalar_metrics': {
                # these are currently dicts; would need to aggregate them to a single number if needed
                # 'unsucc_conn': unsucc_conn,
                # 'lost_conn': lost_conn,
                # num UEs without any connection
                # 'num_ues_wo_conn': sum([1 if len(ue.bs_dr) == 0 else 0 for ue in self.ue_list]),
                # 'avg_utility': np.mean([ue.utility for ue in self.ue_list]),
                'sum_utility': sum([ue.utility for ue in self.ue_list])
            },
            # vector metrics are metrics that contain a dict of values for each step with UE --> metric
            # currently not supported (needs some adjustments in simulator)
            'vector_metrics': {
                'dr': {f'UE {ue}': ue.curr_dr for ue in self.ue_list},
                'utility': {f'UE {ue}': ue.utility for ue in self.ue_list},
            }
        }
        return info_dict

    def step(self, action):
        """
        Environment step consisting of 1) Applying actions, 2) Updating data rates and rewards, 3) Moving UEs,
        4) Updating data rates and rewards again (avg with reward before), 5) Updating the observation

        In the base env here, only one UE applies actions per time step. This is overwritten in other env variants.

        :param action: Action to be applied. Here, for a single UE. In other env variants, for all UEs.
        :return: Tuple of next observation, reward, done, info
        """
        prev_obs = self.obs

        #  get & apply action
        action_dict = self.get_ue_actions(action)
        penalties = self.apply_ue_actions(action_dict)

        # add new UE according to configured interval
        # important to add it here, after the action is applied but before the reward is calculated
        # after action is applied, otherwise a central approach could already apply a decision for the newly added UE
        # before reward is calculated, otherwise the reward and observation keys would not match (required by RLlib)
        if self.new_ue_interval is not None and self.time > 0 and self.time % self.new_ue_interval == 0:
            self.add_new_ue()

        # move UEs, update data rates and rewards in between; increment time
        rewards_before = self.update_ue_drs_rewards(penalties=penalties)
        lost_conn = self.move_ues()
        # penalty of -1 for lost connections due to movement (rather than active disconnect)
        # penalties = {ue: -1 * lost_conn[ue] for ue in self.ue_list}
        # update of drs is needed even if we don't need the reward
        rewards_after = self.update_ue_drs_rewards(penalties=None, update_only=True)
        # rewards = {ue: np.mean([rewards_before[ue], rewards_after[ue]]) for ue in self.ue_list}
        rewards = rewards_before
        self.time += 1

        # get and return next obs, reward, done, info
        self.obs = self.get_obs()
        reward = self.step_reward(rewards)
        done = self.done()
        # dummy unsucc_conn -1 since it's currently not of interest
        # unsucc_conn = {ue: -1 for ue in self.ue_list}
        info = self.info()
        self.log.info("Step", time=self.time, prev_obs=prev_obs, action=action, reward=reward, next_obs=self.obs,
                      done=done)
        return self.obs, reward, done, info

    def render(self, mode='human'):
        """Plot and visualize the current status of the world. Return the patch of actors for animation."""
        # list of matplotlib "artists", which can be used to create animations
        patch = []

        # limit to map borders
        plt.xlim(0, self.map.width)
        plt.ylim(0, self.map.height)

        # users & connections
        # show utility as red to yellow to green. use color map for [0,1) --> normalize utility first
        colormap = cm.get_cmap('RdYlGn')
        norm = plt.Normalize(-20, 20)

        for ue in self.ue_list:
            # plot connections to all BS
            for bs, dr in ue.bs_dr.items():
                color = colormap(norm(log_utility(dr)))
                # add black background/borders for lines to make them better visible if the utility color is too light
                patch.extend(plt.plot([ue.pos.x, bs.pos.x], [ue.pos.y, bs.pos.y], color=color,
                                      path_effects=[pe.SimpleLineShadow(shadow_color='black'), pe.Normal()]))
                                      # path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()]))
            # plot UE
            patch.extend(ue.plot())

        # base stations
        for bs in self.bs_list:
            patch.extend(bs.plot())

        # title isn't redrawn in animation (out of box) --> static --> show time as text inside box, top-right corner
        patch.append(plt.title(type(self).__name__))
        # extra info: time step, total data rate & utility
        patch.append(plt.text(0.9*self.map.width, 0.95*self.map.height, f"t={self.time}"))
        patch.append(plt.text(0.9*self.map.width, 0.9*self.map.height, f"dr={self.total_dr:.2f}"))
        patch.append(plt.text(0.9*self.map.width, 0.85*self.map.height, f"util={self.total_utility:.2f}"))

        # legend doesn't change --> only draw once at the beginning
        # if self.time == 0:
        #     plt.legend(loc='upper left')
        return patch

    def add_new_ue(self, velocity='slow'):
        """Simulate arrival of a new UE in the env"""
        # choose ID based on last UE's ID; assuming it can be cast to int
        id = int(self.ue_list[-1].id) + 1
        # choose a random position on the border of the map for the UE to appear
        pos = self.map.rand_border_point()
        new_ue = User(str(id), self.map, pos.x, pos.y, movement=RandomWaypoint(self.map, velocity=velocity))

        # seed with fixed but unique seed to have different movement
        new_ue.seed(self.env_seed + id * 100)
        # reset to ensure the new seed is applied, eg, to always select the same "random" waypoint with the same seed
        new_ue.reset()
        self.ue_list.append(new_ue)
