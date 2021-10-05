"""
Auxiliary functions for calculating the utility of achieving a certain data rate (for a UE).
Attention: The absolute reward that's achieved with different utilities cannot be compared directly (diff ranges)!
"""
import numpy as np

from deepcomp.util.constants import MIN_UTILITY, MAX_UTILITY


def linear_clipped_utility(curr_dr, max_dr=MAX_UTILITY):
    """
    Utility that directly equals the data rate, increasing linearly up to a given maximum.

    :param max_dr: Maximum data rate at which the utility does not increase further
    :return: Utility
    """
    assert curr_dr >= 0 and max_dr >= 0
    assert MIN_UTILITY == 0 and MAX_UTILITY == max_dr, \
        "The chosen linear utility requires MIN_UTILITY=0 and sensible MAX_UTILITY. Set sensible values manually!"
    return np.clip(curr_dr, MIN_UTILITY, MAX_UTILITY)


def step_utility(curr_dr, req_dr):
    """
    Flat negative utility as long as the required data rate is not met; then positive. Nothing in between.

    :param curr_dr: Current data rate
    :param req_dr: Required data rate
    :return: Min or max utility depending on whether the required data rate is met
    """
    if curr_dr >= req_dr:
        return MAX_UTILITY
    return MIN_UTILITY


def log_utility(curr_dr):
    """
    More data rate increases the utility following a log function: High initial increase, then flattens.

    :param curr_dr: Current data rate
    :param factor: Factor to multiply the log function with
    :param add: Add to current data rate before passing to log function
    :return: Utility
    """
    # 4*log(0.1+x) looks good: around -10 for no dr; 0 for 0.9 dr; slightly positive for more
    # 10*log10(0.1+x) is even better because it's steeper, is exactly -10 for dr=0, and flatter for larger dr
    # with many UEs where each UE only gets around 0.1 data rate, 100*log(0.9+x) looks good (eg, 50 UEs on medium env)

    # better: 10*log10(x) --> clip to [-20, 20]; -20 for <= 0.01 dr; +20 for >= 100 dr
    # ensure min/max utility are set correctly for this utility function
    assert MIN_UTILITY == -20 and MAX_UTILITY == 20, "The chosen log utility requires min/max utility to be -20/+20"
    if curr_dr == 0:
        return MIN_UTILITY
    return np.clip(10 * np.log10(curr_dr), MIN_UTILITY, MAX_UTILITY)
