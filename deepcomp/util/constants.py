"""
Utility module holding constants that are used across different places.
Such that this is the single point in the code to adjust these constants.
"""
import os
import pathlib


# logging settings
LOG_ROUND_DIGITS = 3

# use sets for O(1) include checks
CENTRAL_ALGS = {'ppo', 'random', 'fixed', 'brute-force'}
MULTI_ALGS = {'ppo', '3gpp', 'fullcomp', 'dynamic'}
SUPPORTED_ALGS = CENTRAL_ALGS.union(MULTI_ALGS)
SUPPORTED_AGENTS = {'single', 'central', 'multi'}
SUPPORTED_ENVS = {'small', 'medium', 'large', 'custom'}
SUPPORTED_RENDER = {'html', 'gif', 'both', None}
SUPPORTED_SHARING = {'max-cap', 'resource-fair', 'rate-fair', 'proportional-fair'}
SUPPORTED_REWARDS = {'min', 'sum'}

# small epsilon used in denominator to avoid division by zero
EPSILON = 1e-16

# constants to tune "fairness" of proportional-fair sharing
# high alpha --> closer to max cap; high beta --> closer to resource-fair; alpha = beta = 1 is used in 3G
# actually no, alpha=1=beta converges to exactly the same allocation as resource-fair for stationary users!
# https://en.wikipedia.org/wiki/Proportionally_fair#User_prioritization
FAIR_WEIGHT_ALPHA = 1
FAIR_WEIGHT_BETA = 1


# constants regarding result files
def get_result_dirs(result_dir=None):
    """
    Return the path to the result dir, test dir, and video dir.
    If a custom result dir is provided, use that. Otherwise, default to project root/results.
    """
    if result_dir is None:
        # project root (= repo root; where the readme is) for file access
        _this_dir = pathlib.Path(__file__).parent.absolute()
        project_root = _this_dir.parent.parent.absolute()
        result_dir = os.path.join(project_root, 'results')

    train_dir = os.path.join(result_dir, 'train')
    test_dir = os.path.join(result_dir, 'test')
    video_dir = os.path.join(result_dir, 'videos')

    # create dirs
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    return result_dir, train_dir, test_dir, video_dir
