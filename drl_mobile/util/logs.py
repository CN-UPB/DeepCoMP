import logging

import gym
import numpy as np
import structlog
from structlog.stdlib import LoggerFactory


def config_logging(round_digits):
    """Configure logging using structlog, stdlib logging, and custom FloatRounder to round to spec numb digits"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('drl_mobile').setLevel(logging.WARNING)
    logging.getLogger('drl_mobile.env.simulation').setLevel(logging.DEBUG)
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


class FloatRounder:
    """
    A structlog processor for rounding floats.
    Inspired by: https://github.com/underyx/structlog-pretty/blob/master/structlog_pretty/processors.py
    Extended to also round numbers in lists; recurse nested lists. Less try-except. Handle np.arrays.
    """
    def __init__(self, digits=3, only_fields=None, not_fields=None, np_array_to_list=True):
        """Create a processor that rounds numbers in the event values
        :param digits: The number of digits to round to
        :param only_fields: An iterable specifying the fields to round
        :param not_fields: An iterable specifying fields not to round
        :param np_array_to_list: Whether to cast np.array to list for nicer printing
        """
        self.digits = digits
        self.np_array_to_list = np_array_to_list
        try:
            self.only_fields = set(only_fields)
        except TypeError:
            self.only_fields = None
        try:
            self.not_fields = set(not_fields)
        except TypeError:
            self.not_fields = None

    def _round(self, value):
        """Round floats, unpack lists, convert np.arrays to lists"""
        # round floats
        if isinstance(value, float):
            return round(value, self.digits)
        # convert np.array to list
        if self.np_array_to_list:
            if isinstance(value, np.ndarray):
                return self._round(list(value))
        # round values in lists recursively (to handle lists of lists)
        if isinstance(value, list):
            for idx, item in enumerate(value):
                value[idx] = self._round(item)
            return value
        # return any other values as they are
        return value

    def __call__(self, _, __, event_dict):
        for key, value in event_dict.items():
            if self.only_fields is not None and key not in self.only_fields:
                continue
            if key in self.not_fields:
                continue
            if isinstance(value, bool):
                continue  # don't convert True to 1.0

            event_dict[key] = self._round(value)
        return event_dict
