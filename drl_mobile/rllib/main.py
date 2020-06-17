"""Main execution script for RLlib"""

import logging
import os

import gym
import structlog
from structlog.stdlib import LoggerFactory
import tensorflow as tf
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


from drl_mobile.rllib.env import TunnelEnv, DummyMobileEnv
from drl_mobile.util.logs import FloatRounder


log = structlog.get_logger()


def config_logging(round_digits):
    """Configure logging using structlog, stdlib logging, and custom FloatRounder to round to spec numb digits"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    logging.getLogger('drl_mobile.env.simulation').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    gym.logger.set_level(gym.logger.ERROR)
    # structlog.configure(logger_factory=LoggerFactory())
    structlog.configure(logger_factory=LoggerFactory(),
                        processors=[
                            structlog.stdlib.filter_by_level,
                            # structlog.stdlib.add_logger_name,
                            # structlog.stdlib.add_log_level,
                            # structlog.stdlib.PositionalArgumentsFormatter(),
                            # structlog.processors.StackInfoRenderer(),
                            # structlog.processors.format_exc_info,
                            # structlog.processors.UnicodeDecoder(),
                            # structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                            FloatRounder(digits=round_digits, not_fields=['sinr', 'signal', 'interference']),
                            structlog.dev.ConsoleRenderer()
                            # structlog.stdlib.render_to_log_kwargs,
                            # structlog.processors.JSONRenderer()
                        ])


if __name__ == "__main__":
    # config_logging(round_digits=3)

    ray.init()

    # env config
    env_config = {'len_tunnel': 5, 'len_episode': 10}
    # ray config
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1
    # shorter training for quicker debugging
    config['train_batch_size'] = 200
    # config['env'] = TunnelEnv
    config['env_config'] = env_config

    # train
    trainer = ppo.PPOTrainer(config=config, env=TunnelEnv)
    result = trainer.train()
    print(pretty_print(result))
