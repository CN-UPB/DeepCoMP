"""
Utility module holding constants that are used across different places.
Such that this is the single point in the code to adjust these constants.
"""
# if order doesn't matter, prefer sets for O(1) include checks
SUPPORTED_ALGS = {'ppo', 'greedy-best', 'greedy-all', 'random', 'fixed'}
SUPPORTED_AGENTS = {'single', 'central', 'multi'}
SUPPORTED_ENVS = {'small', 'medium', 'large', 'custom'}
SUPPORTED_RENDER = {'html', 'gif', 'both', None}
SUPPORTED_SHARING = {'max-cap', 'resource-fair', 'rate-fair', 'proportional-fair'}
