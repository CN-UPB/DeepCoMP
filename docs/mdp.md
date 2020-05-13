# MDP Formulation 

## v0.2: Just BS selection, basic radio model (WIP)

* Same as v0, but with path loss, SNR to data rate calculation
  * At the beginning: No interference, no scheduling. Add incrementally
* State/Observation: S = [Achievable data rates per BS, connected BS]?
    * Using achievable dr directly, works very poorly. 
    Suspected reason: Data rates are much larger (up to 150x) than the connected values, such that 
    the agent basically cannot see anymore to which BS it is connected
    * Simply cutting off data rates, eg, at 3 Mbit/s, works much better

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