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
    parser.add_argument('--target-utility', type=int, default=None, help="Target mean sum utility for training")
    parser.add_argument('--continue', type=str, help="Continue training agent at given path (loads last checkpoint")
    parser.add_argument('--separate-agent-nns', action='store_true',
                        help="Only relevant for multi-agent RL. Use separate NNs for each agent instead of sharing.")
    parser.add_argument('--lstm', action='store_true', help="Whether or not to use an LSTM cell")
    # environment
    parser.add_argument('--env', type=str, choices=SUPPORTED_ENVS, default='small', help="Env/Map size")
    parser.add_argument('--bs-dist', type=int, default=100, help="Distance between BS. Only supported by medium env.")
    parser.add_argument('--eps-length', type=int, default=100, help="Number of time steps per episode")
    parser.add_argument('--static-ues', type=int, default=0, help="Number of static UEs in the environment")
    parser.add_argument('--slow-ues', type=int, default=0, help="Number of slow UEs in the environment")
    parser.add_argument('--fast-ues', type=int, default=0, help="Number of fast UEs in the environment")
    # could implement this simply by processing the number and increasing the static, slow, fast UEs correspondingly
    # before passing to env creation
    # parser.add_argument('--mixed-ues', type=int, default=0,
    #                     help="Number of UEs in the environment, equally mixed between different movement speeds")
    parser.add_argument('--sharing', type=str, choices=SUPPORTED_SHARING, default='resource-fair',
                        help="Sharing model used by BS to split resources and/or rate among connected UEs.")
    # evaluation
    parser.add_argument('--rand-train', action='store_true', help="Randomize training episodes.")
    parser.add_argument('--cont-train', action='store_true', help="Continuous training without resetting.")
    parser.add_argument('--rand-test', action='store_true', help="Randomize testing and evaluation episodes.")
    parser.add_argument('--fixed-rand-eval', action='store_true',
                        help="Evaluate once with fixed episodes and then again with random episodes.")
    parser.add_argument('--test', type=str, help="Test trained agent at given path (auto. loads last checkpoint)")
    parser.add_argument('--video', type=str, choices=SUPPORTED_RENDER, default=None,
                        help="How (and whether) to render the testing video.")
    parser.add_argument('--eval', type=int, default=0, help="Number of evaluation episodes after testing")
    parser.add_argument('--seed', type=int, default=None, help="Seed for the RNG (algorithms and environment)")

    args = parser.parse_args()
    log.info('CLI args', args=args)

    assert getattr(args, 'continue') is None or args.test is None, "Use either --continue or --test, not both."
    assert args.rand_test is False or args.fixed_rand_eval is False, "Use either --rand-test or --fixed-rand-eval."
    assert not (args.cont_train and args.rand_train), "Either train continuously or with random episodes."
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
    if args.target_utility is not None:
        stop_criteria['custom_metrics/sum_utility_mean'] = args.target_utility

    # train or load trained agent; only set train=True for ppo agent
    train = args.test is None
    agent_path = None
    if args.test is not None:
        agent_path = os.path.abspath(args.test)
    agent_path_continue = None
    if args_continue is not None:
        agent_path_continue = os.path.abspath(args_continue)

    # create RLlib config (with env inside) & simulator
    config = create_env_config(agent=args.agent, map_size=args.env, bs_dist=args.bs_dist,
                               num_static_ues=args.static_ues, num_slow_ues=args.slow_ues,
                               num_fast_ues=args.fast_ues, sharing_model=args.sharing, eps_length=args.eps_length,
                               num_workers=args.workers, train_batch_size=args.batch_size, seed=args.seed,
                               agents_share_nn=not args.separate_agent_nns, use_lstm=args.lstm,
                               rand_episodes=args.rand_train)

    # for sequential multi agent env
    # config['no_done_at_end'] = True

    # TODO: for continuous setting with fixed horizon
    if args.cont_train:
        config['horizon'] = args.eps_length
        config['soft_horizon'] = True
        config['no_done_at_end'] = True

    # TODO: hyper-param search; probably easiest with simple grid search
    # default ppo params: https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
    # config['entropy_coeff'] = 0.01
    # lr: 5e-5, lr_schedule: None, gae lambda: 1.0, kl_coeff: 0.2
    # config['lr'] = ray.tune.uniform(1e-6, 1e-4)
    # config['gamma'] = ray.tune.uniform(0.9, 0.99)
    # config['lambda'] = ray.tune.uniform(0.7, 1.0)
    # lr_schedule: https://github.com/ray-project/ray/issues/7912#issuecomment-609833914
    # eg, [[0, 0.01], [1000, 0.0001]] will start (t=0) lr=0.01 and linearly decr to lr=0.0001 at t=1000
    # config['lr_schedule'] = [[0, 0.01], [50000, 1e-5]]
    # import hyperopt as hp
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
    # set episode randomization for testing and evaluation according to CLI arg
    sim.run(render=args.video, log_dict=log_dict)

    # evaluate over multiple episodes
    if args.eval > 0:
        sim.run(num_episodes=args.eval, write_results=True)

        # evaluate again with toggled episode randomization if --fixed-rand-eval
        if args.fixed_rand_eval:
            log.info('Evaluating again with toggled episode randomization', rand_episodes=not args.rand_test)
            # set changed testing mode which is then saved to the data frame
            sim.cli_args.rand_test = not args.rand_test
            # make new result filename to avoid overwriting the existing one
            sim.set_result_filename()
            sim.run(num_episodes=args.eval, write_results=True)

    log.info('Finished', agent=agent_path)


if __name__ == '__main__':
    main()
