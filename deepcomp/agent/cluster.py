"""
Functionality to compute static clusters of cells up front, offline for static
clustering.
Inspired by the non-overlapping, static clustering approach by Marsch & Fettweis
https://ieeexplore.ieee.org/document/5963458

Select clusters of cells with size M (configurable) that maximize the mean SINR
to possible UE positions. These positions are unknown here and therefore approx
by uniform positions on the map.
"""

class ClusterBuilder:
    def __init__(self, cluster_size, map, bs_list, ue_list=None):
        self.cluster_size = cluster_size
        self.map = map
        self.bs_list = bs_list
        # TODO: generate static UEs uniformly and then use them to test the clusters

    def eval_cluster(self, cluster):
        """For a given cluster (as dict), evaluate it's mean SINR."""
        # TODO: connect all UEs to their clusters

    def build_clusters(self):
        pass
