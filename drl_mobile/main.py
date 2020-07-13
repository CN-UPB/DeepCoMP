"""Main execution script used for experimentation"""
import logging
import argparse

import structlog

from drl_mobile.util.constants import SUPPORTED_ALGS, SUPPORTED_ENVS, SUPPORTED_AGENTS, SUPPORTED_RENDER
from drl_mobile.util.simulation import Simulation
from drl_mobile.util.logs import config_logging
from drl_mobile.util.env_setup import create_env_config


log = structlog.get_logger()


def setup_cli():
    """Create CLI parser and return parsed args"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--workers', type=int, default=1, help="Number of ray workers")
    parser.add_argument('--eps-length', type=int, default=30, help="Number of time steps per episode")
    parser.add_argument('--train-iter', type=int, default=1, help="Number of training iterations")
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
    parser.add_argument('--no-eval', action='store_true', help="Disable additional evaluation episodes after testing.")

    args = parser.parse_args()
    log.info('CLI args', args=args)
    return args


def main():
    config_logging(round_digits=3)
    args = setup_cli()

    # stop training when any of the criteria is met
    stop_criteria = {
        'training_iteration': args.train_iter,
        # 'episode_reward_mean': 250
    }
    # train or load trained agent; only set train=True for ppo agent
    train = args.test is None
    # name of the RLlib dir to load the agent from for testing
    # agent_path = '../training/PPO/PPO_MultiAgentMobileEnv_0_2020-07-01_15-42-31ypyfzmte/checkpoint_25/checkpoint-25'
    agent_path = f'../training/PPO/{args.test}'
    # seed for agent & env
    seed = 42

    # create RLlib config (with env inside) & simulator
    config = create_env_config(agent=args.agent, map_size=args.env, num_slow_ues=args.slow_ues,
                               num_fast_ues=args.fast_ues, eps_length=args.eps_length,
                               num_workers=args.workers, train_batch_size=args.batch_size, seed=seed)
    sim = Simulation(config=config, agent_name=args.alg, debug=False)

    # train
    if train and args.alg == 'ppo':
        agent_path, analysis = sim.train(stop_criteria)

    # load & test agent
    sim.load_agent(rllib_path=agent_path, rand_seed=seed, fixed_action=[1, 1])

    # simulate one episode and render
    log_dict = {
        'drl_mobile.util.simulation': logging.DEBUG,
        'drl_mobile.env.entities.user': logging.DEBUG,
        # 'drl_mobile.env.entities.station': logging.DEBUG
    }
    sim.run(render=args.video, log_dict=log_dict)

    # evaluate over multiple episodes
    if not args.no_eval:
        sim.run(num_episodes=30)


if __name__ == '__main__':
    main()
