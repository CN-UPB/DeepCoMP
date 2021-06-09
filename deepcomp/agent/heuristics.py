"""
Heuristic algorithms to use as baseline. Only work as multi-agent, not central (would be the same anyways).
"""
import copy
import random

import numpy as np
from shapely.geometry import Point

from deepcomp.agent.base import MultiAgent


class Heuristic3GPP(MultiAgent):
    """
    Agent that is always connected to at most one BS. Greedily chooses the BS with highest achievable data rate.
    This is comparable to 3GPP LTE cell selection based on highest SINR (with a hysteresis threshold of 0)
    """

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


class FullCoMP(MultiAgent):
    """Agent that always greedily connects to all BS. I refer to this agent as 'FullCoMP' in the paper."""

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


class DynamicSelection(MultiAgent):
    """
    Heuristic that dynamically selects cells per UE depending on the SINR.
    It always selects the strongest cell with SINR-1st and all cells that are within epsilon * SINR-1st.
    This represents a configurable intermediate approach between our single-cell 3GPP and the multi-cell Full CoMP approach.
    With epsilon=0, this heuristic equals the Full CoMP approach. With epsilon=1, it equals the Full CoMP approach.

    Based on the following paper: 'Multi-point fairness in resource allocation for C-RAN downlink CoMP transmission'
    https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-015-0501-4
    """
    def __init__(self, epsilon):
        """
        :param epsilon: Scaling factor
        """
        super().__init__()
        self.epsilon = epsilon

    def compute_action(self, obs, policy_id):
        """Select strongest BS and all that are within epsilon * SINR of that BS"""
        # get set of selected cells
        best_snr = max(obs['dr'])
        threshold = best_snr * self.epsilon
        selected_bs = [idx for idx, snr in enumerate(obs['dr']) if snr >= threshold]

        connected_bs = [idx for idx, conn in enumerate(obs['connected']) if conn]
        # disconnect from any BS not in the set of selected BS
        for bs in connected_bs:
            if bs not in selected_bs:
                # 0 = noop --> select BS with BS index + 1
                return bs + 1

        # then connect to BS inside set, starting with the strongest --> sort with decreasing SINR
        selected_bs_sorted = sorted(selected_bs, key=lambda idx: obs['dr'][idx], reverse=True)
        for bs in selected_bs_sorted:
            if not obs['connected'][bs]:
                return bs + 1

        # else do nothing
        return 0


class StaticClustering(MultiAgent):
    """
    Cluster cells into static, non-overlapping groups of fixed size M, which then form a group for CoMP-JT.
    Inspired by the approach by Marsch & Fettweis: https://ieeexplore.ieee.org/document/5963458
    Instead of clustering by solving an ILP for max SINR, here, simply choose closest cells.
    """
    def __init__(self, cluster_size, bs_list, seed=None, clusters=None):
        self.cluster_size = cluster_size
        self.bs_list = bs_list
        self.seed = seed
        # random number generator for clustering approach
        self.rng = random.Random()
        self.rng.seed(seed)
        # build clusters up front, which are then used for online cell selection
        self.clusters = clusters
        if self.clusters is None:
            self.clusters = self.build_clusters()

    def build_clusters(self):
        """
        Take list of cells and build clusters of configured size, choosing the closest neighbors.

        :returns: Dict with cell ID --> set of cell IDs in same cluster
        """
        clusters = dict()
        remaining_bs = copy.copy(self.bs_list)
        curr_cluster = set()

        while len(remaining_bs) > 0:
            # start new cluster with random remaining cell
            if len(curr_cluster) == 0:
                bs = self.rng.choice(remaining_bs)
                curr_cluster.add(bs)
                remaining_bs.remove(bs)

            # add closest cell to cluster
            else:
                center_x = np.mean([bs.pos.x for bs in curr_cluster])
                center_y = np.mean([bs.pos.y for bs in curr_cluster])
                center = Point(center_x, center_y)
                closest_bs = min(remaining_bs, key=lambda x: center.distance(x.pos))
                # add to cluster and remove from remaining cells
                curr_cluster.add(closest_bs)
                remaining_bs.remove(closest_bs)

            # if cluster is full, save and reset
            if len(curr_cluster) == self.cluster_size:
                for bs in curr_cluster:
                    clusters[bs] = curr_cluster
                curr_cluster = set()

        # add remaining cells in curr cluster, even if it's not full
        for bs in curr_cluster:
            clusters[bs] = curr_cluster

        return clusters

    def compute_action(self, obs, policy_id):
        """Select strongest BS and all BS that are in the same cluster. Independent of policy_id."""
        # get set of cells in cluster
        best_bs = self.bs_list[np.argmax(obs['dr'])]
        cluster = self.clusters[best_bs]
        cluster_idx = [self.bs_list.index(bs) for bs in cluster]

        connected_bs_idx = [idx for idx, conn in enumerate(obs['connected']) if conn]
        # disconnect from any BS not in the set of selected BS
        for bs_idx in connected_bs_idx:
            if bs_idx not in cluster_idx:
                # 0 = noop --> select BS with BS index + 1
                return bs_idx + 1

        # then connect to BS inside set, starting with the strongest --> sort with decreasing SINR
        selected_bs_sorted = sorted(cluster_idx, key=lambda idx: obs['dr'][idx], reverse=True)
        for bs_idx in selected_bs_sorted:
            if not obs['connected'][bs_idx]:
                return bs_idx + 1

        # else do nothing
        return 0
