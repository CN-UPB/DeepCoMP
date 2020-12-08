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
MULTI_ALGS = {'ppo', 'greedy-best', 'greedy-all', 'dynamic'}
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

# constants regarding result and trainig files
# project root (= repo root; where the readme is) for file access
_this_dir = pathlib.Path(__file__).parent.absolute()
PROJECT_ROOT = _this_dir.parent.parent.absolute()
RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')
TRAIN_DIR = os.path.join(RESULT_DIR, 'PPO')
TEST_DIR = os.path.join(RESULT_DIR, 'test')
VIDEO_DIR = os.path.join(RESULT_DIR, 'videos')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')


def create_result_dirs():
    """Create directories for saving training, testing results and videos"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


create_result_dirs()
