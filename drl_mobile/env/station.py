import structlog
import numpy as np


class Basestation:
    """A base station sending data to connected UEs"""
    # TODO: optionally disable interference
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        # list of connected UEs
        self.conn_ue = []
        # radius for plotting; should reflect coverage
        # 46m is approx radius for 1mbit with curr settings and no interference
        # TODO: calculate radius automatically based on radio model; can't be calc in closed form but numerically approx
        # TODO: or better visualize decreasing dr somehow
        self.radius = 46
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
        # if disable_interference=True (set by env), ignore interference; just calc SNR, not SINR
        self.disable_interference = False

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

    def received_power(self, distance):
        """Return the received power"""
        return 10**((self.tx_power - self.path_loss(distance)) / 10)

    def interference(self, ue_pos, active_bs):
        """Return interference power at given UE position based on given list of active BS."""
        interfering_bs = [bs for bs in active_bs if bs != self]
        interf_power = 0
        for bs in interfering_bs:
            dist_to_ue = bs.pos.distance(ue_pos)
            interf_power += bs.received_power(dist_to_ue)
        return interf_power

    def sinr(self, ue_pos, active_bs):
        """Return the singal-to-noise-and-interference (SINR) ratio given a UE position and list of active BS"""
        distance = self.pos.distance(ue_pos)
        signal = self.received_power(distance)
        interference = 0
        if not self.disable_interference:
            interference = self.interference(ue_pos, active_bs)
        self.log.debug('SINR to UE', ue_pos=str(ue_pos), active_bs=active_bs, distance=distance,
                       signal=signal, interference=interference, disable_interference=self.disable_interference)
        return signal / (self.noise + interference)

    def data_rate(self, ue_pos, active_bs):
        """Return the achievable data rate for a UE at the given position with given list of active BS."""
        sinr = self.sinr(ue_pos, active_bs)
        dr = self.bw * np.log2(1 + sinr)
        self.log.debug('Data rate to UE', ue_pos=str(ue_pos), active_bs=active_bs, sinr=sinr, dr=dr)
        return dr
