import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from deepcomp.env.single_ue.variants import DatarateMobileEnv, NormDrMobileEnv, RelNormEnv, MaxNormEnv


class MultiAgentMobileEnv(RelNormEnv, MultiAgentEnv):
    """
    Multi-UE and multi-agent env.
    Inherits the parent env's (eg, DatarateMobileEnv) constructor, step, visualization
    & overwrites MultiAgentEnv's reset and step.
    https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    """
    def __init__(self, env_config):
        # this calls parent env.__ini__() since MultiAgentEnv doesn't have an __init__
        super().__init__(env_config)
        # inherits attributes, obs and action space from parent env

        # how to aggregate rewards from multiple UEs (sum or min utility)
        self.reward_agg = env_config['reward']

    def get_ue_actions(self, action):
        """
        Retrieve the action per UE from the RL agent's action and return in in form of a dict.
        Does not yet apply actions to env.

        :param action: Action that depends on the agent type (single, central, multi)
        :return: Dict that consistently (indep. of agent type) maps UE (object) --> action
        """
        # get action for each UE based on ID
        return {ue: action[ue.id] for ue in self.ue_list if ue.id in action}

    def get_obs(self):
        """Return next obs: Dict with UE --> obs"""
        obs = dict()
        for ue in self.ue_list:
            obs[ue.id] = self.get_ue_obs(ue)
        return obs

    def step_reward(self, rewards):
        """
        Return rewards as they are but use UE ID as key instead of UE itself.
        The reward key needs to be same as obs key & sortable not just hashable.
        """
        # sum_rewards = sum(rewards.values())
        # return {ue.id: sum_rewards for ue in rewards.keys()}
        # return {ue.id: r for ue, r in rewards.items()}

        # variant: add aggregated utility of UEs at the same BS
        new_rewards = dict()
        for ue, r in rewards.items():
            # initialize to own utility in case the UE is not connected to any BS and has no neighbors
            agg_util = r
            # neighbors include the UE itself
            neighbors = ue.ues_at_same_bs()
            if len(neighbors) > 0:
                # aggregate utility of different UEs as configured
                if self.reward_agg == 'sum':
                    agg_util = sum([rewards[neighbor] for neighbor in neighbors])
                elif self.reward_agg == 'min':
                    agg_util = min([rewards[neighbor] for neighbor in neighbors])
                else:
                    raise NotImplementedError(f"Unexpected reward aggregation: {self.reward_agg}")
            new_rewards[ue.id] = agg_util
            self.log.debug('Reward', ue=ue, neighbors=neighbors, own_r=r, agg_util=agg_util)
        return new_rewards

    def done(self):
        """Return dict of dones: UE --> done?"""
        done = super().done()
        dones = {ue.id: done for ue in self.ue_list}
        dones['__all__'] = done
        return dones

    def info(self):
        """Return info for each UE as dict. Required by RLlib to be similar to obs."""
        info_dict = super().info()
        return {ue.id: info_dict for ue in self.ue_list}


class SeqMultiAgentMobileEnv(MultiAgentMobileEnv):
    """
    Multi-agent env where all agents observe and act sequentially rather than simultaneously within each time step.
    All agents act sequentially within a single time step before they move and time increments.
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        # order of UEs to make sequential decisions; for now identical to list order
        self.ue_order = self.ue_list
        self.ue_order_idx = 0
        self.curr_ue = self.ue_order[self.ue_order_idx]

    def get_obs(self):
        """Return only obs for current UE, such that only this UE acts"""
        return {self.curr_ue.id: self.get_ue_obs(self.curr_ue)}

    def step_reward(self, rewards):
        """Only reward for current UE. Calc as before"""
        new_rewards = super().step_reward(rewards)
        return {self.curr_ue.id: new_rewards[self.curr_ue.id]}

    def done(self):
        """Set done for current UE. For all when reaching the last UE"""
        done = super().done()
        dones = {
            self.curr_ue.id: done,
            '__all__': done,
        }
        return dones

    def info(self):
        """Same for info: Only for curr UE. Then increment to next UE since it's the last operation in the step"""
        info_dict = super(MultiAgentMobileEnv, self).info()
        return {self.curr_ue.id: info_dict}

    def step(self, action):
        """Overwrite step to do sequential steps per agent without moving UEs and incrementing time in each step"""
        # when reaching the last UE in the order, move time, UEs, etc
        # if self.ue_order_idx >= len(self.ue_order):
        #     self.ue_order_idx = 0
        #     # move UEs, update drs, increment time
        #     self.move_ues()
        #     self.update_ue_drs_rewards(penalties=None, update_only=True)
        #     self.time += 1
        # self.curr_ue = self.ue_order[self.ue_order_idx]

        # same as in normal step
        prev_obs = self.obs
        action_dict = self.get_ue_actions(action)
        penalties = self.apply_ue_actions(action_dict)
        rewards = self.update_ue_drs_rewards(penalties=penalties)

        # increment UE idx to now handle next user; but do not move or increment time
        if self.ue_order_idx + 1 < len(self.ue_order):
            self.ue_order_idx += 1
        else:
            self.ue_order_idx = 0
            # move UEs, update drs, increment time
            self.move_ues()
            self.update_ue_drs_rewards(penalties=None, update_only=True)
            self.time += 1
        self.curr_ue = self.ue_order[self.ue_order_idx]

        self.obs = self.get_obs()
        reward = self.step_reward(rewards)
        done = self.done()
        info = self.info()
        self.log.info("Step", time=self.time, prev_obs=prev_obs, action=action, reward=reward, next_obs=self.obs,
                      done=done)
        return self.obs, reward, done, info
