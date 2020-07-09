"""Auxiliary functions for calculating the utility of achieving a certain data rate (for a UE)"""
import numpy as np


def step_utility(curr_dr, req_dr):
    """
    Flat negative utility as long as the required data rate is not met; then positive. Nothing in between.

    :param curr_dr: Current data rate
    :param req_dr: Required data rate
    :return: Utility (-10 or +10)
    """
    if curr_dr >= req_dr:
        return 10
    return -10


def log_utility(curr_dr):
    """
    More data rate increases the utility following a log function: High initial increase, then flattens.

    :param curr_dr: Current data rate
    :return: Utility
    """
    # 4*log(0.1+x) looks good: around -10 for no dr; 0 for 0.9 dr; slightly positive for more
    return 4 * np.log(0.1 + curr_dr)
