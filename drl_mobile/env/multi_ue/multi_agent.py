import numpy as np
import gym.spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.variants import DatarateMobileEnv


class MultiAgentMobileEnv(MultiAgentEnv, DatarateMobileEnv):
    """
    Multi-UE and multi-agent env.
    Inherits & overwrites MultiAgentEnv's reset and step.
    Inherits DatarateMobileEnv's constructor, visualization, etc.
    https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    """
    def __init__(self, env_config):
        # this calls DatarateMobileEnv.__ini__() since MultiAgentEnv doesn't have an __init__
        super().__init__(env_config)
        # inherits attributes from parent env
        self.observation_space = self.static_obs_space(self.bs_list, self.ue_list, self.dr_cutoff, self.sub_req_dr)
        self.action_space = self.static_action_space(self.bs_list, self.ue_list)

    @staticmethod
    def static_obs_space(bs_list, ue_list, dr_cutoff, sub_req_dr):
        return gym.spaces.Dict({
            ue.id: DatarateMobileEnv.static_obs_space(bs_list, ue_list, dr_cutoff, sub_req_dr) for ue in ue_list
        })

    @staticmethod
    def static_action_space(bs_list, ue_list):
        return gym.spaces.Dict({
            ue.id: DatarateMobileEnv.static_action_space(bs_list, ue_list) for ue in ue_list
        })

    def reset(self):
        """Reset the env and return observations from all UEs"""
        self.time = 0
        for ue in self.ue_list:
            ue.reset()
        for bs in self.bs_list:
            bs.reset()
        # multi-agent: get obs for all UEs and return dict with ue.id --> ue's obs
        self.obs = {ue.id: self.get_obs(ue) for ue in self.ue_list}
        return self.obs

    def step(self, action_dict):
        """
        Apply actions of all agents (here UEs) and step the environment
        :param action_dict: Dict of UE IDs --> selected action
        :return: obs, rewards, dones, infos. Again in the form of dicts: UE ID --> value
        """
        # TODO: avoid duplicate code; write function for joint parts of step in base, central, and multi-agent
        prev_obs = self.obs
        obs = {}
        rewards = {}

        # step all UEs
        for ue in self.ue_list:
            penalty = 0

            # apply action for each UE; 0= noop
            if action_dict[ue.id] > 0:
                bs = self.bs_list[action_dict[ue.id] - 1]
                # penalty of -3 for unsuccessful connection attempt
                penalty -= 3 * (not ue.connect_to_bs(bs, disconnect=True))

            # move and calc reward for UE
            reward_before = self.calc_reward(ue, penalty)
            num_lost_conn = ue.move()
            # add penalty of -1 for each lost connection through movement (rather than actively disconnected)
            penalty -= num_lost_conn
            reward_after = self.calc_reward(ue, penalty)
            rewards[ue.id] = np.mean([reward_before, reward_after])

            # next obs
            obs[ue.id] = self.get_obs(ue)

        self.time += 1

        # return next obs, reward, done, infos (after incrementing time!)
        self.obs = obs
        # done and info are the same for all UEs
        done = self.time >= self.episode_length
        dones = {ue.id: done for ue in self.ue_list}
        dones['__all__'] = done
        infos = {ue.id: {'time': self.time} for ue in self.ue_list}
        self.log.info("Step", time=self.time, prev_obs=prev_obs, action=action_dict, rewards=rewards, next_obs=self.obs, done=done)
        return self.obs, rewards, dones, infos

# TODO: implement similar variant with total current dr as in the central env
