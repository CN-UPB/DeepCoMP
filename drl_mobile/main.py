"""Main execution script used for experimentation"""
import logging
import argparse

import structlog

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

    args = parser.parse_args()
    log.info('CLI args', args=args)
    return args


def main():
    config_logging(round_digits=3)
    args = setup_cli()

    # settings
    # stop training when any of the criteria is met
    stop_criteria = {
        'training_iteration': args.train_iter,
        # 'episode_reward_mean': 250
    }
    # train or load trained agent; only set train=True for ppo agent
    train = False
    agent_name = 'random'
    # name of the RLlib dir to load the agent from for testing
    agent_path = '../training/PPO/PPO_MultiAgentMobileEnv_0_2020-07-01_15-42-31ypyfzmte/checkpoint_25/checkpoint-25'
    # seed for agent & env
    seed = 42

    # create RLlib config (with env inside) & simulator
    config = create_env_config(eps_length=args.eps_length, num_workers=args.workers, train_batch_size=args.batch_size,
                               seed=seed)
    sim = Simulation(config=config, agent_name=agent_name, debug=False)

    # train
    if train:
        agent_path, analysis = sim.train(stop_criteria)

    # load & test agent
    sim.load_agent(rllib_path=agent_path, rand_seed=seed, fixed_action=[1, 1])

    # simulate one episode and render
    log_dict = {
        'drl_mobile.util.simulation': logging.DEBUG,
        # 'drl_mobile.env.entities': logging.DEBUG
    }
    sim.run(render='video', log_dict=log_dict)

    # evaluate over multiple episodes
    # sim.run(num_episodes=30)


if __name__ == '__main__':
    main()
