import ray.rllib.agents.ppo as ppo

from drl_mobile.rllib.env import TunnelEnv, DummyMobileEnv


# dummy function with hard-coded env to get sth simple working
def create_rllib_agent(seed=None, train=True):
    if train:
        config = ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 1
        config['seed'] = seed
        # shorter training for faster debugging
        config['train_batch_size'] = 200
        # config['log_level'] = 'INFO'    # default: warning
        # in case of RLlib env is the env_config
        # TODO: avoid hard-coding env here
        config['env_config'] = {'len_tunnel': 5, 'len_episode': 10}
        return ppo.PPOTrainer(config=config, env=TunnelEnv)
    else:   # TODO: rllib testing
        raise NotImplementedError('Still have to implement testing with RLlib')