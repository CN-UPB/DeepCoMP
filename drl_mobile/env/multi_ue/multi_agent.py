from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.variants import DatarateMobileEnv


class MultiAgentMobileEnv(DatarateMobileEnv, MultiAgentEnv):
    """
    Multi-UE and multi-agent env.
    Inherits DatarateMobileEnv's step & overwrites MultiAgentEnv's reset and step.
    Inherits DatarateMobileEnv's constructor, stepping, visualization, etc.
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

    def apply_ue_actions(self, action):
        """
        Apply actions of all UEs.

        :param: Dict of actions: UE --> action
        :return: Dict of penalties for each UE based on unsuccessful connection attempts (-3)
        """
        penalties = dict()

        # apply action: try to connect to BS; or: 0 = no op
        for ue in self.ue_list:
            penalties[ue] = 0

            # apply action for UE; 0= noop
            if action[ue.id] > 0:
                bs = self.bs_list[action[ue.id] - 1]
                # penalty of -3 for unsuccessful connection attempt
                penalties[ue] -= 3 * (not ue.connect_to_bs(bs, disconnect=True))

        return penalties

    def next_obs(self):
        """Return next obs: Dict with UE --> obs"""
        obs = dict()
        for ue in self.ue_list:
            obs[ue.id] = self.get_obs(ue)
        return obs

    def step_reward(self, rewards):
        """Return rewards as they are: Dict with UE --> reward"""
        return rewards

    def done(self):
        """Return dict of dones: UE --> done?"""
        done = self.time >= self.episode_length
        dones = {ue.id: done for ue in self.ue_list}
        dones['__all__'] = done
        return dones

    def info(self):
        """Return info for each UE as dict"""
        info_dict = super().info()
        return {ue.id: info_dict for ue in self.ue_list}
