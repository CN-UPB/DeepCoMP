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
    Hence, there is no cinterference between BSs.
* We do not consider assignment of RBs explicitly, but assume that
    * BS assign all RBs to connected users, ie, transmit as much data rate as possible
    * It's configurable how the data rate is shared among connected UEs: See below
* Based on the SNR and the number of connected users at a BS, I calculate the achievable data rate per UE from a BS
* UEs can connect to multiple BS and their data rates add up
* UEs can only connect to BS that are not too far away, ie, where SNR is above a fixed threshold   

### Radio sharing model

BS support different, configurable sharing models: Max. capacity, resource-fair, or rate-fair sharing.

#### Max. capacity

Maximize capacity by assigning full data rate to the UE with highest achievable data rate. 0 to all other UEs.

![max_cap](gifs/max_cap.gif)

#### Resource-fair sharing

Split RBs equally among all connected UEs. Assume infinitely many and small RBs. Results in different data rates based on distance/path loss: `dr_nominal = dr / num_ues`

![res_fair](gifs/resource_fair.gif)

#### Rate-fair sharing (current default)

Ensure same data rate for all connected UEs. Connecting far-away UEs becomes very expensive and decreases total capacity and rate for all connected UEs.

![rate_fair](gifs/rate_fair.gif)

#### Proportional-fair sharing (TODO)

Balance capacity maximization and resource-fair sharing, by calculating a priority for each UE: [Wikipedia](https://en.wikipedia.org/wiki/Proportionally_fair#User_prioritization).
Split RBs proportional to that priority. Tune fairness using weights alpha and beta (between extremes max. cap. and resource-fair).


### Todo

See HK's mail from 22.06.:

* *Done*: Current time-wise fair sharing is fine. But volume-wise would be better (Wifi). Or even better proportional fair.
* Assuming a high frequency reuse factor such that neighboring BS do not interfere is like GSM and outdated. I should consider a stand-alone scheduler (greedy?) at some point instead.
    * Or control power or RB/channel assignment by RL like in paper below
* Assuming that UEs can receive from multiple BS at multiple frequencies at the same time may not be realistic. Not sure what is?
* *Done*: Allowing UEs to connect to BS that offer 1/10 the required rate doesn't make sense, eg, if the required rate is very high. Instead: S * factor c > N? With configurable c.


Model considerations after reading recent paper (26.06.):

* What do I optimize? Should I also just optimize sum of all UE data rates? Wouldn't that lead to exploitation of best UEs and starvation of remaining?
* Co-channel interference + power control or sub-channel/RB assignment?
* Add UE positions (or distances?) to observations?
* UE movement following Brownian motion?
