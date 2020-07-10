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


class GreedyAllSelection:
    """Agent that always greedily connects to all BS."""
    # TODO: extension: only to those that exceed a min dr threshold.
    #  to avoid constantly tryign to connect to BS that are out of range --> leads to penalty

    def compute_action(self, obs, policy_id):
        """
        Compute action for a UE. Try to connect to all BS. Prioritize BS with higher data rate.

        :param obs: Observations of the UE
        :param policy_id: Ignored
        :return: Action for the UE
        """
        # identify BS that are not yet connected
        disconn_bs = [idx for idx, conn in enumerate(obs['connected']) if not conn]
        # if connected to all BS already, do nothing
        if len(disconn_bs) == 0:
            return 0
        # else connect to the BS with the highest data rate
        best_bs = disconn_bs[0]
        best_dr = obs['dr'][best_bs]
        for bs in disconn_bs:
            if obs['dr'][bs] > best_dr:
                best_bs = bs
                best_dr = obs['dr'][bs]
        # 0 = noop --> select BS with BS index + 1
        return best_bs + 1
