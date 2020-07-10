"""
Heuristic algorithms to use as baseline. Only work as multi-agent, not central (would be the same anyways).
"""
import numpy as np


class GreedyBestSelection:
    """Agent that is always connected to at most one BS. Greedily chooses the BS with highest achievable data rate."""
    def compute_action(self, obs, policy_id):
        """
        Compute an action for one UE by connecting to the BS with highest data rate (if not connected yet).
        Gets called for all UEs by simulator.

        :param obs: Observation of a UE
        :param policy_id: Ignored since the heuristic behaves identically for all UEs; just based on obs.
        :return: Selected action: 0 = noop. 1-n = index of BS +1 to connect/disconnect
        """
        # identify BS with highest data rate; in case of a tie, take the first one
        best_bs = np.argmax(obs['dr'])
        # if already connected to this BS, stay connected = do nothing
        if obs['connected'][best_bs]:
            return 0
        # if connected to other BS, disconnect first
        if sum(obs['connected']) > 0:
            conn_bs = obs['connected'].index(1)
            return conn_bs + 1
        # else: not connected yet --> connect to best BS
        return best_bs + 1

