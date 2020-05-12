import structlog
import numpy as np


class Basestation:
    """A base station sending data to connected UEs"""
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        # list of connected UEs
        self.conn_ue = []
        # radius for plotting; should reflect coverage
        # 45m is approx radius for 1mbit with curr settings and no interference
        # TODO: calculate radius automatically based on radio model; can't be calc in closed form but numerically approx
        # TODO: or better visualize decreasing dr somehow
        self.radius = 45
        self.coverage = pos.buffer(self.radius)
        # set constants for SINR and data rate calculation
        # numbers originally from https://sites.google.com/site/lteencyclopedia/lte-radio-link-budgeting-and-rf-planning
        # changed numbers to get shorter range --> simulate smaller map
        self.bw = 9e6   # in Hz?
        self.frequency = 2500    # in MHz
        self.noise = 1e-9   # in mW
        self.tx_power = 30  # in dBm (was 40)
        self.height = 50    # in m
        # just consider downlink for now; more interesting for most apps anyways

        self.log = structlog.get_logger(id=self.id, pos=str(self.pos))

    def __repr__(self):
        return self.id

    @property
    def active(self):
        """The BS is active iff it's connected to at least 1 UE"""
        return len(self.conn_ue) > 0

    def path_loss(self, distance, ue_height=1.5):
        """Return path loss in dBm to a UE at a given position. Calculation using Okumura Hata, suburban indoor"""
        ch = 0.8 + (1.1 * np.log10(self.frequency) - 0.7) * ue_height - 1.56 * np.log10(self.frequency)
        const1 = 69.55 + 26.16 * np.log10(self.frequency) - 13.82 * np.log10(self.height) - ch
        const2 = 44.9 - 6.55 * np.log10(self.height)
        return const1 + const2 * np.log10(distance)

    def snr(self, path_loss=0):
        """Return the singal-to-noise (SNR) ratio given a certain path loss"""
        signal = 10**((self.tx_power - path_loss) / 10)
        return signal / self.noise

    def data_rate(self, ue_pos, sinr_threshold=1e-5):
        """Return the achievable data rate for a UE at the given position. 0 if below SINR threshold"""
        distance = self.pos.distance(ue_pos)
        path_loss = self.path_loss(distance)
        snr = self.snr(path_loss)
        # TODO: add interference
        dr = self.bw * np.log2(1 + snr)
        self.log.debug('Data rate to UE', ue_pos=str(ue_pos), distance=distance, path_loss=path_loss, snr=snr, dr=dr)
        return dr
