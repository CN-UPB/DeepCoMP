import structlog
import numpy as np


# SNR threshold required for UEs to connect to this BS. This threshold corresponds roughly to a distance of 69m.
SNR_THRESHOLD = 2e-8


class Basestation:
    """A base station sending data to connected UEs"""
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.num_conn_ues = 0

        # set constants for SINR and data rate calculation
        # numbers originally from https://sites.google.com/site/lteencyclopedia/lte-radio-link-budgeting-and-rf-planning
        # changed numbers to get shorter range --> simulate smaller map
        self.bw = 9e6   # in Hz?
        self.frequency = 2500    # in MHz
        self.noise = 1e-9   # in mW
        self.tx_power = 30  # in dBm (was 40)
        self.height = 50    # in m
        # just consider downlink for now; more interesting for most apps anyways

        # for visualization: circles around BS that show connection range (69m) and 1 Mbit range (46m); no interference
        self.range_conn = pos.buffer(69)
        self.range_1mbit = pos.buffer(46)

        # FIXME: enabling logging still shows deepcopy error. See https://github.com/hynek/structlog/issues/268
        # TODO: log num conn ues
        # self.log = structlog.get_logger(id=self.id, pos=str(self.pos))

    def __repr__(self):
        return str(self.id)

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

    def data_rate_unshared(self, ue):
        """
        Return the achievable data rate for a given UE assuming that it gets the BS' full, unshared data rate.
        :param ue: UE requesting the achievable data rate
        :return: Return the max. achievable data rate for the UE if it were/is connected to the BS.
        """
        snr = self.snr(ue.pos)
        dr_ue_unshared = self.bw * np.log2(1 + snr)
        return dr_ue_unshared

    def data_rate_shared(self, ue, dr_ue_unshared, sharing_model='resource-fair'):
        """
        Return the shared data rate the given UE would get based on its unshared data rate and a sharing model.
        :param ue: UE requesting the achievable data rate
        :param dr_ue_unshared: The UE's unshared achievable data rate
        :param sharing_model: A model for sharing rate/resources among connected UEs.
            Currently supported: 'resource-fair'
        :return: The UE's final, shared data rate that it (could/does) get from this BS
        """
        supported_models = ('resource-fair')
        assert sharing_model in supported_models, f"{sharing_model=} not supported. {supported_models=}"
        dr_ue_shared = None

        # resource fair = time/bandwidth-fair: split time slots/bandwidth/RBs equally among all connected UEs
        if sharing_model == 'resource-fair':
            # split data rate by all already connected UEs + this UE if it is not connected yet
            split_by = self.num_conn_ues
            if self not in ue.conn_bs:
                # what would be the data rate if this UE connects as well?
                split_by += 1
            # nominal data rate depends on sharing model
            dr_ue_shared = dr_ue_unshared / split_by

        return dr_ue_shared

    def data_rate(self, ue):
        """
        Return the achievable data rate for a given UE (may or may not be connected already).
        Share & split the achievable data rate among all connected UEs, pretending this UE is also connected.
        :param ue: UE requesting the achievable data rate
        :return: Return the max. achievable data rate for the UE if it were/is connected to the BS.
        """
        # achievable data rate if it wasn't shared with any other connected UEs
        dr_ue_unshared = self.data_rate_unshared(ue)
        # final, shared data rate depends on sharing model
        dr_ue_shared = self.data_rate_shared(ue, dr_ue_unshared)
        print(f"bs={self.id}, {dr_ue_unshared=}, {dr_ue_shared=}")
        # self.log.debug('Achievable data rate', ue=ue.id, snr=snr, dr_ue=dr_ue, dr_ue_shared=dr_ue_shared,
        # split_by=split_by)
        return dr_ue_shared

    # TODO: without interference, this really just translates to a fixed distance. so decide based on distance instead?
    #  would be simpler & faster
    def can_connect(self, ue_pos):
        """Return if a UE at a given pos can connect to this BS. That's the case if its SNR is above a threshold."""
        can_connect = self.snr(ue_pos) > SNR_THRESHOLD
        # TODO: replace with logging once it works again
        print(f"bs={self.id}, {ue_pos=}, {can_connect=}")
        return can_connect
