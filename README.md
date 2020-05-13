# deep-rl-mobility-management

Simple simulation of mobility management scenario to use with deep RL

## Setup

Should work with Python 3.6+. Tested with Python 3.7. Tensorflow 1 doesn't work on Python 3.8.

```
pip install -r requirements
```

## Todos

* (Make decisions in fixed decision interval of XY simulator steps)
* Add schedules/RBs to radio model?
* As soon as centralized works: Move to distributed RL. Even before adding offloading.

## Findings

13.05.2020:

* Binary observations: [BS available?, BS connected?] work very well
* Replacing binary "BS available?" with achievable data rate by BS does not work at all
* Probably, because data rate is magnitudes larger (up to 150x) than "BS connected?" --> agent becomes blind to 2nd part of obs
* Just cutting the data rate off at some small value (eg, 3 Mbit/s) leads to much better results
* Agent keeps trying to connect to all BS, even if out of range. --> Subtracting req. dr by UE + higher penalty (both!) solves the issue
