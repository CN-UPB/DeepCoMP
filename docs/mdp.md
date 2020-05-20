# MDP Formulation 

## v0.3: Multi-UE-BS selection, basic radio model (WIP)

* Multiple moving UEs, each selecting to which BS to connect. 
* Simple radio load model: Split achievable load equally among connected UEs

* Actions:
    * For now: Make actions and movement for each UE in different time step
    * When just applying a single agent for both UEs, it works really bad and even diverges.
    Likely because the experienced obs-action-reward-obs tuples don't make any sense: 
    After obs and choosing action for UE1, it does not get next obs for UE1, but next obs for UE2.
    Otherwise, it would not know the obs for making action for UE2.
        * Adding the current UE's ID to the obs does not help and still leads to divergence.

## [v0.2](https://github.com/CN-UPB/deep-rl-mobility-management/releases/tag/v0.2): Just BS selection, basic radio model (week 21)

* Same as v0, but with path loss, SNR to data rate calculation. No interference or scheduling yet.
* State/Observation: S = [Achievable data rates per BS (processed), connected BS]
    * Using achievable dr directly, works very poorly. 
    Suspected reason: Data rates are much larger (up to 150x) than the connected values, such that 
    the agent basically cannot see anymore to which BS it is connected
    * Simply cutting off data rates, eg, at 3 Mbit/s, works much better. Problem: Where to cut off?
    * First subtracting the required data rate from the achievalbe data rate helps a lot!
    It changes the observation to be negative if the data rate doesn't suffice.
    * What works best is auto clipping and normalization:
        1. Subtract the required data rate --> Obs. negative if data rate too small
        1. Cut off data rate at req data rate --> Obs. range now [-req_dr, +req_dr].
        1. Normalize by dividing by req. data rate --> Obs. range now [-1, 1]
* Action space as before: Select a BS to connect/disconnect in each time step for the single UE

Example: PPO with auto clipping & normalization observations after 10k training

![v0.2 example](gifs/v02.gif)

## [v0.1](https://github.com/CN-UPB/deep-rl-mobility-management/releases/tag/v0.1): Just BS selection, no radio model (week 19)

Env. dynamics:

* Only one moving UE. Multiple (2) BS. UE selects to which BS(s) to connect
* No radio model yet, instead, each BS has a fixed range, which it covers. No capacities yet.
* Worked with an out-of-the box RL agent (PPO algorithm)

Observations: 

* Binary vector of |S|=2x num BS with S = [available BS, connected BS].
* Eg. S = [1, 1, 0, 1] means that both BS are available (in range) but the UE currently is only connected to the 2nd BS.

Action: 

* 1 action/decision in each time step to connect or disconnect from BS
* Action space: {0, 1, ..., num BS}. 0 = no op. = keep everything as is. 1+ = if not connected to BS i yet, connect (try); if connected already, disconnect

Reward:

* +10 for each time step in which UE is connected to at least one BS
* -10 for each time step in which it isn't
* -1 if connection attempt failed (agent tried to connect to BS that's out of reach)