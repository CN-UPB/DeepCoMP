import logging

import gym
import structlog
from structlog.stdlib import LoggerFactory
from structlog_round import FloatRounder

from drl_mobile.util.constants import LOG_ROUND_DIGITS


# TODO: also log to file (optionally)
def config_logging():
    """Configure logging using structlog, stdlib logging, and custom FloatRounder to round to spec numb digits"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    logging.getLogger('drl_mobile.util.simulation').setLevel(logging.INFO)
    # logging.getLogger('drl_mobile.env.util.movement').setLevel(logging.DEBUG)
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
                            FloatRounder(digits=LOG_ROUND_DIGITS, not_fields=['sinr', 'signal', 'interference']),
                            structlog.dev.ConsoleRenderer()
                            # structlog.stdlib.render_to_log_kwargs,
                            # structlog.processors.JSONRenderer()
                        ])
