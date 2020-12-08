"""Multi-UE envs with single, centralized agent controlling all UEs at once."""
import gym.spaces

from deepcomp.env.single_ue.base import MobileEnv
from deepcomp.env.single_ue.variants import NormDrMobileEnv, DatarateMobileEnv, RelNormEnv, MaxNormEnv


class CentralBaseEnv(MobileEnv):
    """
    Abstract base env for central multi-UE coordination.
    Variants need to inherit first from this class, 2nd from the single-UE base env.
    Other than that, they only need to define the obs space accordingly.
    All methods are adjusted to centralized by this wrapper.
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        # how to aggregate rewards from multiple UEs (sum or min utility)
        self.reward_agg = env_config['reward']

        # if the number of UEs varies over time, the action space needs to be large enough to control all UEs
        if self.max_ues > self.num_ue:
            self.log.warning("Num. UEs varies over time. Setting action and observation space for max. number of UEs.",
                             curr_num_ue=self.num_ue, max_ues=self.max_ues)

        # observation space to be defined in sub classes!
        # actions: FOR EACH UE: select a BS to be connected to/disconnect from or noop
        self.action_space = gym.spaces.MultiDiscrete([self.num_bs + 1 for _ in range(self.max_ues)])

    # overwrite modular functions used within step that are different in the centralized case
    def get_obs(self):
        """Get obs for all UEs based on the parent environment's obs definition"""
        obs = {key: [] for key in self.observation_space.spaces.keys()}
        for ue in self.ue_list:
            # get properly normalized observations from NormDrMobileEnv for each UE
            ue_obs = super().get_ue_obs(ue)
            # extend central obs
            for key in self.observation_space.spaces.keys():
                # handle Discrete or Binary obs --> single number
                if type(ue_obs[key]) == int:
                    obs[key].append(ue_obs[key])
                # remaining Box or MultiDiscrete/Binary obs --> vector/list
                else:
                    obs[key].extend(ue_obs[key])

        # to support variable numbers of UEs, pad observations with zeros for all non-existing UEs
        # since the observations need a fixed length, which is set to the max. number of UEs
        if self.num_ue < self.max_ues:
            num_ue_missing = self.max_ues - self.num_ue
            for obs_name, obs_space in self.observation_space.spaces.items():
                # get the number of obs per UE by looking at the shape and dividing it by num UEs
                assert obs_space.shape[0] % self.max_ues == 0, f"{obs_name} does not have a fixed number of obs per UE"
                obs_per_ue = obs_space.shape[0] / self.max_ues
                # append zeros for the missing UEs
                obs[obs_name].extend([0 for _ in range(int(num_ue_missing * obs_per_ue))])

        return obs

    def get_ue_actions(self, action):
        """Apply action. Here: Actions for all UEs. Return unsuccessful connection attempts."""
        assert self.action_space.contains(action), f"Action {action} does not fit action space {self.action_space}"
        # get action for each UE based on index
        return {ue: action[i] for i, ue in enumerate(self.ue_list)}

    def step_reward(self, rewards):
        """Return aggregated reward of all UEs as step reward"""
        if self.reward_agg == 'sum':
            return sum(rewards.values())
        if self.reward_agg == 'min':
            return min(rewards.values())
        raise NotImplementedError(f"Unexpected reward aggregation: {self.reward_agg}")


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
        if self.dist_obs:
            obs_space['dist'] = gym.spaces.Box(low=0, high=1, shape=(self.num_ue * self.num_bs,))
        if self.next_dist_obs:
            obs_space['next_dist'] = gym.spaces.Box(low=0, high=1, shape=(self.num_ue * self.num_bs,))

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
            'connected': gym.spaces.MultiBinary(self.num_ue * self.num_bs),
            # 'can_connect': gym.spaces.MultiBinary(self.num_ue * self.num_bs),
            # 'num_conn': gym.spaces.MultiDiscrete([self.num_bs + 1 for _ in range(self.num_ue)]),
            # 'ues_at_bs': gym.spaces.MultiDiscrete([self.num_ue+1 for _ in range(self.num_bs)]),
            # 'ues_at_bs': gym.spaces.Box(low=0, high=1, shape=(self.num_bs,)),
            'dr_total': gym.spaces.Box(low=0, high=1, shape=(self.num_ue,)),
            # 'unshared_dr': gym.spaces.Box(low=0, high=self.dr_cutoff, shape=(self.num_ue * self.num_bs,)),
        }
        self.observation_space = gym.spaces.Dict(obs_space)

    # only overwrite this to only include num UEs per BS just once in obs instead of repeating it multiple times
    # def get_obs(self):
    #     # get basic obs from parent
    #     obs = super().get_obs()
    #     # only include first part of ues_at_bs obs; remove repetitions
    #     obs['ues_at_bs'] = obs['ues_at_bs'][:self.num_bs]
    #     return obs


class CentralRelNormEnv(CentralBaseEnv, RelNormEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        # use max. number of UEs instead of actual number to support varying numbers of UEs
        obs_space = {
            'connected': gym.spaces.MultiBinary(self.max_ues * self.num_bs),
            'dr': gym.spaces.Box(low=0, high=1, shape=(self.max_ues * self.num_bs,)),
            'utility': gym.spaces.Box(low=-1, high=1, shape=(self.max_ues,)),
        }
        self.observation_space = gym.spaces.Dict(obs_space)


class CentralMaxNormEnv(CentralBaseEnv, MaxNormEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        # use max. number of UEs instead of actual number to support varying numbers of UEs
        obs_space = {
            'connected': gym.spaces.MultiBinary(self.max_ues * self.num_bs),
            'dr': gym.spaces.Box(low=-1, high=1, shape=(self.max_ues * self.num_bs,)),
            'utility': gym.spaces.Box(low=-1, high=1, shape=(self.max_ues,)),
        }
        self.observation_space = gym.spaces.Dict(obs_space)
