from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.base import MobileEnv
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
        # inherits attributes, observations, and actions from parent env

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
        raise NotImplementedError('TODO')
