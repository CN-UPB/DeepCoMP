from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.variants import DatarateMobileEnv, NormDrMobileEnv


class MultiAgentMobileEnv(NormDrMobileEnv, MultiAgentEnv):
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

    def apply_ue_actions(self, action):
        """
        Apply actions of all UEs.

        :param: Dict of actions: UE --> action
        :return: Dict of for each UE based on unsuccessful connection attempts
        """
        unsucc_conn = dict()

        # apply action: try to connect to BS; or: 0 = no op
        for ue in self.ue_list:
            unsucc_conn[ue] = 0

            # apply action for UE; 0= noop
            if action[ue.id] > 0:
                bs = self.bs_list[action[ue.id] - 1]
                unsucc_conn[ue] = not ue.connect_to_bs(bs, disconnect=True)

        return unsucc_conn

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
        return {ue.id: r for ue, r in rewards.items()}

    def done(self):
        """Return dict of dones: UE --> done?"""
        done = self.time >= self.episode_length
        dones = {ue.id: done for ue in self.ue_list}
        dones['__all__'] = done
        return dones

    def info(self, unsucc_conn, lost_conn):
        """Return info for each UE as dict"""
        info_dict = super().info(unsucc_conn, lost_conn)
        return {ue.id: info_dict for ue in self.ue_list}
