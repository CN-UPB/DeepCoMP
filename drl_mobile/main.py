"""Main execution script used for experimentation"""
import os
import argparse
import logging

import structlog
import ray.tune

from drl_mobile.util.constants import SUPPORTED_ALGS, SUPPORTED_ENVS, SUPPORTED_AGENTS, SUPPORTED_RENDER, \
    SUPPORTED_SHARING
from drl_mobile.util.simulation import Simulation
from drl_mobile.util.logs import config_logging
from drl_mobile.util.env_setup import create_env_config


log = structlog.get_logger()


def setup_cli():
    """Create CLI parser and return parsed args"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required args
    parser.add_argument('--agent', type=str, choices=SUPPORTED_AGENTS, required=True,
                        help="Whether to use a single agent for 1 UE, a central agent, or multi agents")
    # algorithm & training
    parser.add_argument('--alg', type=str, choices=SUPPORTED_ALGS, default='ppo', help="Algorithm")
    parser.add_argument('--workers', type=int, default=1, help="Number of workers for RLlib training and evaluation")
    parser.add_argument('--batch-size', type=int, default=4000, help="Number of training iterations per training batch")
    parser.add_argument('--train-steps', type=int, default=None, help="Max. number of training time steps (if any)")
    parser.add_argument('--train-iter', type=int, default=None, help="Max. number of training iterations (if any)")
    parser.add_argument('--target-reward', type=int, default=None, help="Target mean episode reward for training")
    parser.add_argument('--continue', type=str, help="Continue training agent at given path (loads last checkpoint")
    parser.add_argument('--separate-agent-nns', action='store_true',
                        help="Only relevant for multi-agent RL. Use separate NNs for each agent instead of sharing.")
    # environment
    parser.add_argument('--env', type=str, choices=SUPPORTED_ENVS, default='small', help="Env/Map size")
    parser.add_argument('--eps-length', type=int, default=100, help="Number of time steps per episode")
    parser.add_argument('--slow-ues', type=int, default=0, help="Number of slow UEs in the environment")
    parser.add_argument('--fast-ues', type=int, default=0, help="Number of fast UEs in the environment")
    parser.add_argument('--sharing', type=str, choices=SUPPORTED_SHARING, default='resource-fair',
                        help="Sharing model used by BS to split resources and/or rate among connected UEs.")
    # evaluation
    parser.add_argument('--test', type=str, help="Test trained agent at given path (auto. loads last checkpoint)")
    parser.add_argument('--video', type=str, choices=SUPPORTED_RENDER, default='html',
                        help="How (and whether) to render the testing video.")
    parser.add_argument('--eval', type=int, default=0, help="Number of evaluation episodes after testing")
    parser.add_argument('--seed', type=int, default=None, help="Seed for the RNG (algorithms and environment)")

    args = parser.parse_args()
    log.info('CLI args', args=args)

    assert getattr(args, 'continue') is None or args.test is None, "Use either --continue or --test, not both."
    return args


def main():
    config_logging()
    args = setup_cli()
    # can't use args.continue: https://stackoverflow.com/a/63266666/2745116
    args_continue = getattr(args, 'continue')

    # stop training when any of the criteria is met
    stop_criteria = dict()
    if args.train_steps is not None:
        stop_criteria['timesteps_total'] = args.train_steps
    if args.train_iter is not None:
        stop_criteria['training_iteration'] = args.train_iter
    if args.target_reward is not None:
        stop_criteria['episode_reward_mean'] = args.target_reward

    # train or load trained agent; only set train=True for ppo agent
    train = args.test is None
    agent_path = None
    if args.test is not None:
        agent_path = os.path.abspath(args.test)
    agent_path_continue = None
    if args_continue is not None:
        agent_path_continue = os.path.abspath(args_continue)

    # create RLlib config (with env inside) & simulator
    config = create_env_config(agent=args.agent, map_size=args.env, num_slow_ues=args.slow_ues,
                               num_fast_ues=args.fast_ues, sharing_model=args.sharing, eps_length=args.eps_length,
                               num_workers=args.workers, train_batch_size=args.batch_size, seed=args.seed,
                               agents_share_nn=not args.separate_agent_nns)

    # TODO: for continuous setting with fixed horizon
    # config['horizon'] = args.eps_length
    # config['soft_horizon'] = True
    # config['no_done_at_end'] = True

    # TODO: hyper-param search; probably easiest with simple grid search
    # default ppo params: https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
    # lr: 5e-5, lr_schedule: None, gae lambda: 1.0, kl_coeff: 0.2
    # config['lr'] = ray.tune.uniform(1e-6, 1e-4)
    # config['gamma'] = ray.tune.uniform(0.9, 0.99)
    # config['lambda'] = ray.tune.uniform(0.7, 1.0)
    # lr_schedule: https://github.com/ray-project/ray/issues/7912#issuecomment-609833914
    # eg, [[0, 0.01], [1000, 0.0001]] will start (t=0) lr=0.01 and linearly decr to lr=0.0001 at t=1000
    # config['lr_schedule'] = [[0, 0.01], [50000, 1e-5]]
    import hyperopt as hp
    # from ray.tune.suggest.hyperopt import HyperOptSearch
    # hyperopt = HyperOptSearch(metric='episode_reward_mean', mode='max')

    # add cli args to the config for saving inputs
    sim = Simulation(config=config, agent_name=args.alg, cli_args=args, debug=False)

    # train
    if train and args.alg == 'ppo':
        agent_path, analysis = sim.train(stop_criteria, restore_path=agent_path_continue)

    # load & test agent
    sim.load_agent(rllib_dir=agent_path, rand_seed=args.seed, fixed_action=[1, 1], explore=False)

    # simulate one episode and render
    log_dict = {
        'drl_mobile.util.simulation': logging.DEBUG,
        # 'drl_mobile.env.entities.user': logging.DEBUG,
        # 'drl_mobile.env.entities.station': logging.DEBUG
    }
    sim.run(render=args.video, log_dict=log_dict)

    # evaluate over multiple episodes
    if args.eval > 0:
        sim.run(num_episodes=args.eval, write_results=True)


if __name__ == '__main__':
    main()
