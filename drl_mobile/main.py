"""Main execution script used for experimentation"""
import logging
import argparse

import structlog

from drl_mobile.util.constants import SUPPORTED_ALGS, SUPPORTED_ENVS, SUPPORTED_AGENTS, SUPPORTED_RENDER, TRAIN_DIR
from drl_mobile.util.simulation import Simulation
from drl_mobile.util.logs import config_logging
from drl_mobile.util.env_setup import create_env_config


log = structlog.get_logger()


def setup_cli():
    """Create CLI parser and return parsed args"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--workers', type=int, default=1, help="Number of ray workers")
    parser.add_argument('--eps-length', type=int, default=30, help="Number of time steps per episode")
    parser.add_argument('--train-steps', type=int, default=None, help="Max. number of training time steps (if any)")
    parser.add_argument('--train-iter', type=int, default=None, help="Max. number of training iterations (if any)")
    parser.add_argument('--target-reward', type=int, default=None, help="Target mean episode reward for training")
    parser.add_argument('--batch-size', type=int, default=1000, help="Number of training iterations per training batch")
    parser.add_argument('--alg', type=str, choices=SUPPORTED_ALGS, default='ppo', help="Algorithm")
    parser.add_argument('--agent', type=str, choices=SUPPORTED_AGENTS, required=True,
                        help="Whether to use a single agent for 1 UE, a central agent, or multi agents")
    parser.add_argument('--env', type=str, choices=SUPPORTED_ENVS, default='small', help="Env/Map size")
    parser.add_argument('--slow-ues', type=int, default=0, help="Number of slow UEs in the environment")
    parser.add_argument('--fast-ues', type=int, default=0, help="Number of fast UEs in the environment")
    parser.add_argument('--test', type=str, help="Do not train, only test trained agent at given path (to checkpoint)")
    # parser.add_argument('--cont-train', type=str, help="Load agent from given (checkpoint) path and continue training.")
    parser.add_argument('--video', type=str, choices=SUPPORTED_RENDER, default='html',
                        help="How (and whether) to render the testing video.")
    parser.add_argument('--eval', type=int, default=30, help="Number of evaluation episodes after testing")
    parser.add_argument('--seed', type=int, default=None, help="Seed for the RNG (algorithms and environment)")

    args = parser.parse_args()
    log.info('CLI args', args=args)
    return args


def main():
    config_logging(round_digits=3)
    args = setup_cli()

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
    # name of the RLlib dir to load the agent from for testing; when training always loads the just trained agent
    agent_path = f'{TRAIN_DIR}/{args.test}'

    # create RLlib config (with env inside) & simulator
    config = create_env_config(agent=args.agent, map_size=args.env, num_slow_ues=args.slow_ues,
                               num_fast_ues=args.fast_ues, eps_length=args.eps_length,
                               num_workers=args.workers, train_batch_size=args.batch_size, seed=args.seed)
    # add cli args to the config for saving inputs
    sim = Simulation(config=config, agent_name=args.alg, cli_args=args, debug=False)

    # train
    if train and args.alg == 'ppo':
        agent_path, analysis = sim.train(stop_criteria)

    # load & test agent
    sim.load_agent(rllib_path=agent_path, rand_seed=args.seed, fixed_action=[1, 1])

    # simulate one episode and render
    log_dict = {
        # 'drl_mobile.util.simulation': logging.DEBUG,
        # 'drl_mobile.env.entities.user': logging.DEBUG,
        # 'drl_mobile.env.entities.station': logging.DEBUG
    }
    sim.run(render=args.video, log_dict=log_dict)

    # evaluate over multiple episodes
    if args.eval > 0:
        sim.run(num_episodes=args.eval, write_results=True)


if __name__ == '__main__':
    main()
