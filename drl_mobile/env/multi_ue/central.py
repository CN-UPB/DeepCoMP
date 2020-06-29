"""Multi-UE envs with single, centralized agent controlling all UEs at once."""
import gym.spaces
import numpy as np

from drl_mobile.env.single_ue.base import MobileEnv


class CentralMultiUserEnv(MobileEnv):
    """
    Env where all UEs move, observe and act at all time steps, controlled by a single central agent.
    Otherwise similar to DatarateMobileEnv with auto dr_cutoff and sub_req_dr.
    """
    def __init__(self, env_config):
        """Similar to DatarateMobileEnv but with multi-UEs controlled at once and fixed dr_cutoff, sub_req_dr"""
        super().__init__(env_config)
        # observations: FOR EACH UE: binary vector of BS availability (in range & free cap) + already connected BS
        # 1. Achievable data rate for given UE for all BS (normalized to [-1, 1]) --> Box;
        dr_low = np.full(shape=(self.num_ue * self.num_bs,), fill_value=-1)
        dr_high = np.ones(self.num_ue * self.num_bs)
        # 2. Connected BS --> MultiBinary
        self.observation_space = gym.spaces.Dict({
            'dr': gym.spaces.Box(low=dr_low, high=dr_high),
            'connected': gym.spaces.MultiBinary(self.num_ue * self.num_bs)
        })

        # actions: FOR EACH UE: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.MultiDiscrete([self.num_bs + 1 for _ in range(self.num_ue)])

    def get_obs(self):
        """Observation: Available data rate + connected BS - FOR ALL UEs --> no ue arg"""
        bs_dr = []
        conn_bs = []
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
            ue_conn_bs = [int(bs in ue.conn_bs) for bs in self.bs_list]
            conn_bs.extend(ue_conn_bs)
        return {'dr': bs_dr, 'connected': conn_bs}

    def calc_reward(self, penalty):
        """Calc reward for ALL UEs, similar to normal MobileEnv"""
        reward = penalty
        for ue in self.ue_list:
            reward += super().calc_reward(ue, penalty=0)
        return reward

    def reset(self):
        """Reset environment: Reset all UEs, BS"""
        self.time = 0
        for ue in self.ue_list:
            ue.reset()
        for bs in self.bs_list:
            bs.reset()
        self.obs = self.get_obs()
        return self.obs

    def step(self, action):
        """
        Do 1 time step: Apply action of all UEs and update their position.
        :param action: Array of actions to be applied for each UE
        :return: next observation, reward, done, info
        """
        penalty = 0
        prev_obs = self.obs

        # apply action for each UE; 0 = noop
        for i, ue in enumerate(self.ue_list):
            if action[i] > 0:
                bs = self.bs_list[action[i] - 1]
                # penalty of -3 for unsuccessful connection attempt
                penalty -= 3 * (not ue.connect_to_bs(bs, disconnect=True))

        # move all UEs
        # check connections and reward before and after moving; then avg
        # TODO: usually before & after are the same anyways; so I can drop this if the simulator becomes too slow
        reward_before = self.calc_reward(penalty)
        for ue in self.ue_list:
            num_lost_conn = ue.move()
            # add penalty of -1 for each lost connection through movement (rather than actively disconnected)
            penalty -= num_lost_conn
        reward_after = self.calc_reward(penalty)

        self.time += 1

        # return next observation, reward, done, info
        self.obs = self.get_obs()
        reward = np.mean([reward_before, reward_after])
        done = self.time >= self.episode_length
        info = {'time': self.time}
        self.log.info("Step", time=self.time, prev_obs=prev_obs, action=action, reward_before=reward_before,
                      reward_after=reward_after, reward=reward, next_obs=self.obs, done=done)
        return self.obs, reward, done, info


class CentralRemainingDrEnv(CentralMultiUserEnv):
    """
    Variant of the central multi-agent environment with an additional observation indicating if a UE's rate is
    fulfilled by combining all its connections or not.
    """
    def __init__(self, env_config):
        """Create multi-UE env. Here, just with a slightly extended observation space"""
        super().__init__(env_config)
        # same observations as in CentralMultiUserEnv + extra obs
        # 1. Achievable data rate for given UE for all BS (normalized to [-1, 1]) --> Box;
        dr_low = np.full(shape=(self.num_ue * self.num_bs,), fill_value=-1)
        dr_high = np.ones(self.num_ue * self.num_bs)
        # 2. dr total per UE over all connected BS: 0 = exactly fulfilled, -1 = not fulfilled at all,
        # 1 = twice as much as need
        dr_total_low = np.full(shape=(self.num_ue), fill_value=-1)
        dr_total_high = np.ones(self.num_ue)
        # 3. binary connected BS as before

        self.observation_space = gym.spaces.Dict({
            'dr': gym.spaces.Box(low=dr_low, high=dr_high),
            'dr_total': gym.spaces.Box(low=dr_total_low, high=dr_total_high),
            'connected': gym.spaces.MultiBinary(self.num_ue * self.num_bs)
        })

    def get_obs(self):
        obs_dict = super().get_obs()
        # extend by new observation
        total_dr_list = []
        for ue in self.ue_list:
            total_dr = sum(bs.data_rate(ue) for bs in ue.conn_bs)
            # process by subtracting dr_req, clipping to [-dr_req, dr_req], normalizing to [-1, 1]
            total_dr -= ue.dr_req
            total_dr = min(total_dr, ue.dr_req)
            total_dr = total_dr / ue.dr_req
            total_dr_list.append(total_dr)

        obs_dict['dr_total'] = total_dr_list
        return obs_dict