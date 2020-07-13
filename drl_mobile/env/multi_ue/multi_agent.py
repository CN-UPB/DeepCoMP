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
        self.ues_at_bs_obs = env_config['ues_at_bs_obs']
        # inherits attributes, obs and action space from parent env

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
        obs = dict()
        rewards = dict()
        penalties = dict()

        # step all UEs: 1) apply actions, 2) update dr and reward, 3) move, 4) update dr and reward, 5) get next obs
        # 1) apply actions of all UEs
        for ue in self.ue_list:
            penalties[ue] = 0

            # apply action for UE; 0= noop
            if action_dict[ue.id] > 0:
                bs = self.bs_list[action_dict[ue.id] - 1]
                # penalty of -3 for unsuccessful connection attempt
                penalties[ue] -= 3 * (not ue.connect_to_bs(bs, disconnect=True))

        # 2) update data rates and reward
        for ue in self.ue_list:
            ue.update_curr_dr()
            rewards[ue] = self.calc_reward(ue, penalties[ue])

        # 3) move UEs and update penalty
        for ue in self.ue_list:
            num_lost_conn = ue.move()
            # add penalty of -1 for each lost connection through movement (rather than actively disconnected)
            # TODO: instead of -= I now used =; may affect reward in the end
            penalties[ue] = num_lost_conn

        # 4) update data rates and reward (avg with reward before); get new observation
        for ue in self.ue_list:
            ue.update_curr_dr()
            reward_after = self.calc_reward(ue, penalties[ue])
            rewards[ue] = np.mean([rewards[ue], reward_after])

        # 5) get next obs next obs
        for ue in self.ue_list:
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
