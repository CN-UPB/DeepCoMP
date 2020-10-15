import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.variants import DatarateMobileEnv, NormDrMobileEnv, RelNormEnv


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

        # TODO: test sequential obs and actions
        self.ue_order = self.ue_list
        self.curr_ue_idx = 0

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
        # for ue in self.ue_list:
        #     obs[ue.id] = self.get_ue_obs(ue)
        # TODO: test sequential obs and actions
        ue = self.ue_order[self.curr_ue_idx]
        obs[ue.id] = self.get_ue_obs(ue)
        self.curr_ue_idx = (self.curr_ue_idx + 1) % len(self.ue_order)
        return obs

    def step_reward(self, rewards):
        """
        Return rewards as they are but use UE ID as key instead of UE itself.
        The reward key needs to be same as obs key & sortable not just hashable.
        """
        # return {ue.id: r for ue, r in rewards.items()}

        # variant: add avg reward/utility of all UEs to each UE's own utility
        # avg_reward = np.mean(list(rewards.values()))
        # return {ue.id: 0.5 * r + 0.5 * avg_reward for ue, r in rewards.items()}

        # variant: sum of rewards from all agents
        # total_reward = sum(rewards.values())
        # return {ue.id: total_reward for ue in rewards.keys()}

        # variant: add avg utility of UEs at the same BS
        new_rewards = dict()
        for ue, r in rewards.items():
            neighbors = ue.ues_at_same_bs()
            if len(neighbors) > 0:
                # get the normalized utility = reward for each neighbor
                # avg_util = np.mean([rewards[neighbor] for neighbor in neighbors])
                avg_util = min([rewards[neighbor] for neighbor in neighbors])
            else:
                # if there are no neighbors, then just use own utility/reward
                avg_util = r
            new_r = 0 * r + 1 * avg_util
            self.log.debug('Reward', ue=ue, neighbors=neighbors, own_r=r, avg_util=avg_util, new_r=new_r)
            new_rewards[ue.id] = new_r
        return new_rewards

    def done(self):
        """Return dict of dones: UE --> done?"""
        done = self.time >= self.episode_length
        dones = {ue.id: done for ue in self.ue_list}
        dones['__all__'] = done
        return dones

    def info(self, unsucc_conn, lost_conn):
        """Return info for each UE as dict. Required by RLlib to be similar to obs."""
        info_dict = super().info(unsucc_conn, lost_conn)
        return {ue.id: info_dict for ue in self.ue_list}
