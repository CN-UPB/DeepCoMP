from ray.rllib.env.multi_agent_env import MultiAgentEnv

from drl_mobile.env.single_ue.variants import DatarateMobileEnv, NormDrMobileEnv


class MultiAgentMobileEnv(NormDrMobileEnv, MultiAgentEnv):
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

    def get_ue_actions(self, action):
        """
        Retrieve the action per UE from the RL agent's action and return in in form of a dict.
        Does not yet apply actions to env.

        :param action: Action that depends on the agent type (single, central, multi)
        :return: Dict that consistently (indep. of agent type) maps UE (object) --> action
        """
        # get action for each UE based on ID
        return {ue: action[ue.id] for ue in self.ue_list}

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
