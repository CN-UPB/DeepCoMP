class FloatRounder:
    """
    A structlog processor for rounding floats.
    Inspired by: https://github.com/underyx/structlog-pretty/blob/master/structlog_pretty/processors.py
    Adapted to also round numbers in lists. Less try-except.
    """
    def __init__(self, digits=3, only_fields=None):
        """Create a processor that rounds numbers in the event values
        :param digits: The number of digits to round to
        :param only_fields: An iterable specifying the fields to round
        """
        self.digits = digits
        try:
            self.only_fields = set(only_fields)
        except TypeError:
            self.only_fields = None

    def __call__(self, _, __, event_dict):
        for key, value in event_dict.items():
            if self.only_fields is not None and key not in self.only_fields:
                continue
            if isinstance(value, bool):
                continue  # don't convert True to 1.0

            if isinstance(value, float):
                event_dict[key] = round(value, self.digits)
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, float):
                        event_dict[key][idx] = round(item, self.digits)
        return event_dict
