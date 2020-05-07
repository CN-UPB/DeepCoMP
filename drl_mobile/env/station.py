import structlog
import numpy as np


class Basestation:
    """A base station sending data to connected UEs"""
    def __init__(self, id, pos, cap, radius):
        self.id = id
        self.pos = pos
        self.cap = cap
        self.radius = radius
        self.coverage = pos.buffer(radius)
        # set constants for SINR and data rate calculation
        # numbers from https://sites.google.com/site/lteencyclopedia/lte-radio-link-budgeting-and-rf-planning
        # TODO: adjust/improve (see computeDatarate.py)
        self.bw = 9*10**6   # in Hz?
        self.frequency = 2*10**3    # in MHz
        self.noise = 10**(-95/10)   # in mW
        self.tx_power = 40  # in dBm
        self.height = 50    # in m

        self.log = structlog.get_logger(id=self.id, pos=str(self.pos))

    def __repr__(self):
        return self.id

    def path_loss(self, ue_pos, ue_height=1.5):
        """Return path loss to a UE at a given position. Calculation using Okumura Hata, suburban indoor"""
        ch = 0.8 + (1.1 * np.log10(self.frequency) - 0.7) * ue_height - 1.56 * np.log10(self.frequency)
        const1 = 69.55 + 26.16 * np.log10(self.frequency) - 13.82 * np.log10(self.height) - ch
        const2 = 44.9 - 6.55 * np.log10(self.height)
        distance = self.pos.distance(ue_pos)
        return const1 + const2 * np.log10(distance)

    def snr(self, path_loss=0):
        """Return the singal-to-noise (SNR) ratio given a certain path loss"""
        signal = 10**((self.tx_power - path_loss) / 10)
        return signal / self.noise

    def data_rate(self, ue_pos):
        """Return the achievable data rate for a UE at the given position"""
        path_loss = self.path_loss(ue_pos)
        snr = self.snr(path_loss)
        # TODO: add interference
        dr = self.bw * np.log2(1 + snr)
        self.log.debug('Data rate to UE', ue_pos=ue_pos, path_loss=path_loss, snr=snr, dr=dr)
        return dr
