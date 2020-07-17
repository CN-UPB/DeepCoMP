"""Multi-UE envs with single, centralized agent controlling all UEs at once."""
import gym.spaces
import numpy as np

from drl_mobile.env.single_ue.base import MobileEnv


# TODO: adjust to obs space in single & multi!
class CentralMultiUserEnv(MobileEnv):
    """
    Env where all UEs move, observe and act at all time steps, controlled by a single central agent.
    Otherwise similar to DatarateMobileEnv with auto dr_cutoff and sub_req_dr.
    """
    def __init__(self, env_config):
        """Similar to DatarateMobileEnv but with multi-UEs controlled at once and fixed dr_cutoff, sub_req_dr"""
        super().__init__(env_config)
        self.curr_dr_obs = env_config['curr_dr_obs']
        self.ues_at_bs_obs = env_config['ues_at_bs_obs']

        # observations: FOR EACH UE: vector of BS dr + already connected BS + optionally: total curr dr per UE
        obs_space = {}
        # 1. Achievable data rate for given UE for all BS (normalized to [-1, 1]) --> Box;
        obs_space['dr'] = gym.spaces.Box(low=-1, high=1, shape=(self.num_ue * self.num_bs,))
        # 2. Connected BS --> MultiBinary
        obs_space['connected'] = gym.spaces.MultiBinary(self.num_ue * self.num_bs)
        # 3. Total curr. dr for each UE summed up over all BS connection. Normalized to [-1,1]. Optional
        if self.curr_dr_obs:
            obs_space['dr_total'] = gym.spaces.Box(low=-1, high=1, shape=(self.num_ue,))
        # 4. number of connected UEs per BS --> help distribute UEs better. Optional
        if self.ues_at_bs_obs:
            # at each BS 0 up to all UEs can be connected (no normalization yet)
            obs_space['ues_at_bs'] = gym.spaces.MultiDiscrete([self.num_ue+1 for _ in range(self.num_bs)])

        self.observation_space = gym.spaces.Dict(obs_space)

        # actions: FOR EACH UE: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.MultiDiscrete([self.num_bs + 1 for _ in range(self.num_ue)])

    def get_obs(self):
        """Observation: Available data rate + connected BS (+ total curr dr) - FOR ALL UEs --> no ue arg"""
        bs_dr = []
        conn_bs = []
        total_dr = []
        for ue in self.ue_list:
            # subtract req_dr and auto clip & normalize to [-1, 1]
            ue_bs_dr = []
            for bs in self.bs_list:
                dr_sub = bs.data_rate(ue) - ue.dr_req
                dr_clip = min(dr_sub, ue.dr_req)        # clipped to range [-dr_req, dr_req]
                dr_norm = dr_clip / ue.dr_req
                ue_bs_dr.append(dr_norm)
            bs_dr.extend(ue_bs_dr)
            # connected BS
            ue_conn_bs = [int(bs in ue.bs_dr.keys()) for bs in self.bs_list]
            conn_bs.extend(ue_conn_bs)
            # total curr data rate over all BS
            if self.curr_dr_obs:
                ue_total_dr = ue.curr_dr
                # process by subtracting dr_req, clipping to [-dr_req, dr_req], normalizing to [-1, 1]
                ue_total_dr -= ue.dr_req
                ue_total_dr = min(ue_total_dr, ue.dr_req)
                ue_total_dr = ue_total_dr / ue.dr_req
                total_dr.append(ue_total_dr)

        obs = {'dr': bs_dr, 'connected': conn_bs}
        if self.curr_dr_obs:
            obs['dr_total'] = total_dr
        if self.ues_at_bs_obs:
            obs['ues_at_bs'] = [bs.num_conn_ues for bs in self.bs_list]

        return obs

    # overwrite modular functions used within step that are different in the centralized case
    def apply_ue_actions(self, action):
        """Apply action. Here: Actions for all UEs. Return unsuccessful connection attempts."""
        assert self.action_space.contains(action), f"Action {action} does not fit action space {self.action_space}"
        unsucc_conn = {ue: 0 for ue in self.ue_list}

        # apply action: try to connect to BS; or: 0 = no op
        for i, ue in enumerate(self.ue_list):
            if action[i] > 0:
                bs = self.bs_list[action[i] - 1]
                unsucc_conn[ue] = not ue.connect_to_bs(bs, disconnect=True)

        return unsucc_conn

    def next_obs(self):
        return self.get_obs()

    def step_reward(self, rewards):
        """Return sum of all UE rewards as step reward"""
        return sum(rewards.values())
