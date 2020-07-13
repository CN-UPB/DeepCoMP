"""
Auxiliary functions for calculating the utility of achieving a certain data rate (for a UE).
Attention: The absolute reward that's achieved with different utilities cannot be compared directly (diff ranges)!
"""
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


def log_utility(curr_dr, factor=4, add=0.1):
    """
    More data rate increases the utility following a log function: High initial increase, then flattens.

    :param curr_dr: Current data rate
    :param factor: Factor to multiply the log function with
    :param add: Add to current data rate before passing to log function
    :return: Utility
    """
    # 4*log(0.1+x) looks good: around -10 for no dr; 0 for 0.9 dr; slightly positive for more
    # with many UEs where each UE only gets around 0.1 data rate, 100*log(0.9+x) looks good (eg, 50 UEs on medium env)
    return factor * np.log(add + curr_dr)
