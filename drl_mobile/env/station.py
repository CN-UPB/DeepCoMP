import structlog
import numpy as np


# SNR threshold required for UEs to connect to this BS. This threshold corresponds roughly to a distance of 70m.
SNR_THRESHOLD = 2e-8


class Basestation:
    """A base station sending data to connected UEs"""
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.num_conn_ues = 0
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

        # FIXME: enabling logging still shows deepcopy error. See https://github.com/hynek/structlog/issues/268
        # TODO: log num conn ues
        # self.log = structlog.get_logger(id=self.id, pos=str(self.pos))

    def __repr__(self):
        return self.id

    def reset(self):
        """Reset BS to having no connected UEs"""
        self.num_conn_ues = 0

    def path_loss(self, distance, ue_height=1.5):
        """Return path loss in dBm to a UE at a given position. Calculation using Okumura Hata, suburban indoor"""
        ch = 0.8 + (1.1 * np.log10(self.frequency) - 0.7) * ue_height - 1.56 * np.log10(self.frequency)
        const1 = 69.55 + 26.16 * np.log10(self.frequency) - 13.82 * np.log10(self.height) - ch
        const2 = 44.9 - 6.55 * np.log10(self.height)
        return const1 + const2 * np.log10(distance)

    def received_power(self, distance):
        """Return the received power at a given distance"""
        return 10**((self.tx_power - self.path_loss(distance)) / 10)

    def snr(self, ue_pos):
        """Return the signal-to-noise (SNR) ratio given a UE position."""
        distance = self.pos.distance(ue_pos)
        signal = self.received_power(distance)
        # self.log.debug('SNR to UE', ue_pos=str(ue_pos), distance=distance, signal=signal)
        print(f"SNR: bs={self.id}, {distance=}, {signal=}, {self.noise=}")
        return signal / self.noise

    def data_rate(self, ue):
        """
        Return the achievable data rate for a given UE (may or may not be connected already).
        Split the achievable data rate equally among all connected UEs, pretending this UE is also connected.
        :param ue: UE requesting the achievable data rate
        :return: Return the max. achievable data rate for the UE if it were/is connected to the BS.
        """
        snr = self.snr(ue.pos)
        total_dr = self.bw * np.log2(1 + snr)
        print(f"bs={self.id}, {snr=}, {total_dr=}")
        # split data rate by all already connected UEs + this UE if it is not connected yet
        split_by = self.num_conn_ues
        if self not in ue.conn_bs:
            # what would be the data rate if this UE connects as well?
            split_by += 1
        ue_dr = total_dr / split_by
        # self.log.debug('Achievable data rate', ue=ue.id, sinr=sinr, total_dr=total_dr, ue_dr=ue_dr, split_by=split_by)
        return ue_dr

    # TODO: use this instead of the UE's "can connect"
    def can_connect(self, ue_pos):
        """Return if a UE at a given pos can connect to this BS. That's the case if its SNR is above a threshold."""
        return self.snr(ue_pos) > SNR_THRESHOLD
