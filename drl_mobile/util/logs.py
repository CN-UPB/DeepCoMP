import numpy as np


class FloatRounder:
    """
    A structlog processor for rounding floats.
    Inspired by: https://github.com/underyx/structlog-pretty/blob/master/structlog_pretty/processors.py
    Adapted to also round numbers in lists. Less try-except.
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

    def __call__(self, _, __, event_dict):
        for key, value in event_dict.items():
            if self.only_fields is not None and key not in self.only_fields:
                continue
            if key in self.not_fields:
                continue
            if isinstance(value, bool):
                continue  # don't convert True to 1.0

            # convert np.array to list
            if self.np_array_to_list:
                if isinstance(value, np.ndarray):
                    value = list(value)
                    event_dict[key] = value

            # round
            if isinstance(value, float):
                event_dict[key] = round(value, self.digits)
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, float):
                        event_dict[key][idx] = round(item, self.digits)
        return event_dict
