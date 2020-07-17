"""Multi-UE envs with single, centralized agent controlling all UEs at once."""
import gym.spaces

from drl_mobile.env.single_ue.base import MobileEnv
from drl_mobile.env.single_ue.variants import NormDrMobileEnv, DatarateMobileEnv


class CentralBaseEnv(MobileEnv):
    """
    Abstract base env for central multi-UE coordination.
    Variants need to inherit first from this class, 2nd from the single-UE base env.
    Other than that, they only need to define the obs space accordingly.
    All methods are adjusted to centralized by this wrapper.
    """
    def __init__(self, env_config):
        super().__init__(env_config)

        # observation space to be defined in sub classes!
        # actions: FOR EACH UE: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.MultiDiscrete([self.num_bs + 1 for _ in range(self.num_ue)])

    # overwrite modular functions used within step that are different in the centralized case
    def get_obs(self):
        """Get obs for all UEs based on the parent environment's obs definition"""
        obs = {key: [] for key in self.observation_space.spaces.keys()}
        for ue in self.ue_list:
            # get properly normalized observations from NormDrMobileEnv for each UE
            ue_obs = super().get_ue_obs(ue)
            # extend central obs
            for key in self.observation_space.spaces.keys():
                obs[key].extend(ue_obs[key])
        return obs

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

    def step_reward(self, rewards):
        """Return sum of all UE rewards as step reward"""
        return sum(rewards.values())


class CentralDrEnv(CentralBaseEnv, DatarateMobileEnv):
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
        # actions are defined in the parent class; they are the same for all central envs


class CentralNormDrEnv(CentralBaseEnv, NormDrMobileEnv):
    """
    Variant of the central UE env with different observations (dr is normalized differently).
    Inherits apply actions and action space from central base env, but get_obs from NormDrEnv
    (bc not implemented in central base env);
    """
    def __init__(self, env_config):
        # get action space and env basics from parent
        super().__init__(env_config)
        # same obs as NormDrMobileEnv, just for each UE
        # clip & normalize data rates according to utility.
        # we clip utility at +20, which is reached for a dr of 100
        self.dr_cutoff = 100
        obs_space = {
            'dr': gym.spaces.Box(low=0, high=1, shape=(self.num_ue * self.num_bs,)),
            'connected': gym.spaces.MultiBinary(self.num_ue * self.num_bs)
        }
        self.observation_space = gym.spaces.Dict(obs_space)
