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
        self.curr_dr_obs = env_config['curr_dr_obs']

        # observations: FOR EACH UE: vector of BS dr + already connected BS + optionally: total curr dr per UE
        obs_space = {}
        # 1. Achievable data rate for given UE for all BS (normalized to [-1, 1]) --> Box;
        obs_space['dr'] = gym.spaces.Box(low=-1, high=1, shape=(self.num_ue * self.num_bs,))
        # 2. Connected BS --> MultiBinary
        obs_space['connected'] = gym.spaces.MultiBinary(self.num_ue * self.num_bs)
        # 3. Total curr. dr for each UE summed up over all BS connection. Normalized to [-1,1]. Optional
        if self.curr_dr_obs:
            obs_space['dr_total'] = gym.spaces.Box(low=-1, high=1, shape=(self.num_ue,))

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
            ue_conn_bs = [int(bs in ue.conn_bs) for bs in self.bs_list]
            conn_bs.extend(ue_conn_bs)
            # total curr data rate over all BS
            if self.curr_dr_obs:
                ue_total_dr = sum(bs.data_rate(ue) for bs in ue.conn_bs)
                # process by subtracting dr_req, clipping to [-dr_req, dr_req], normalizing to [-1, 1]
                ue_total_dr -= ue.dr_req
                ue_total_dr = min(ue_total_dr, ue.dr_req)
                ue_total_dr = ue_total_dr / ue.dr_req
                total_dr.append(ue_total_dr)

        if self.curr_dr_obs:
            return {'dr': bs_dr, 'connected': conn_bs, 'dr_total': total_dr}
        return {'dr': bs_dr, 'connected': conn_bs}

    def calc_reward(self, penalty):
        """Calc reward summed up for ALL UEs, similar to normal MobileEnv"""
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
