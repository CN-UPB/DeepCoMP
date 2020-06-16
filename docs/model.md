# Environment Model

* Moving UEs
* Fixed base stations (BSs)
* Actions in every time step

## Radio Model

Radio model mostly implemented in [`drl_mobile/env/station.py`](https://github.com/CN-UPB/deep-rl-mobility-management/blob/master/drl_mobile/env/station.py).

* Focus so far just on downstream, not upstream traffic
* All BS have the same fixed bandwidth, frequency, noise, TX power, and height.
* Based on these and the distance to a UE, the path loss is calculated following the Okumura Hata model for suburban indoor.
* Based on the path loss the SNR for each UE can be calculated
* We assume no interference between UEs and BS based on the following assumptions:
    * BS assign different resource blocks (RB) to different UEs (different time or frequency).
    Hence, there is no interference between UEs at the same BS.
    * Neighboring BS do not use the same RBs but have slightly shifted frequencies.
    Hence, there is no interference between BSs.
* We do not consider assignment of RBs explicitly, but assume that
    * BS assign all RBs to connected users, ie, transmit as much data rate as possible
    * RBs (and thus achievable data rate) are split equally among connected UEs
    * FIXME: That's not what I'm doing [here](https://github.com/CN-UPB/deep-rl-mobility-management/blob/master/drl_mobile/env/station.py#L87).
    I split the achievable data rate *per UE* equally. But that's not the same thing as splitting RBs, is it?
        * Actually, on second thought why not? Wouldn't 50% RBs lead to 50% achievable data rate for each UE?
        * That achievable data rate may still differ for different UEs depending on their position.
* Based on the SNR and the number of connected users at a BS, I calculate the achievable data rate per UE from a BS
* UEs can connect to multiple BS and their data rates add up
    * TODO: UEs can only connect to BS that are not too far away, eg, where the SNR is above a threshold
    * Currently, UEs can only connect if a BS can serve their full rate requirement. That's not what I want.
    ([Code](https://github.com/CN-UPB/deep-rl-mobility-management/blob/master/drl_mobile/env/user.py#L120))
