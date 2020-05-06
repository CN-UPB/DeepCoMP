# deep-rl-mobility-management

Simple simulation of mobility management scenario to use with deep RL

## Setup

Should work with Python 3.6+. Tested with Python 3.7. Tensorflow 1 doesn't work on Python 3.8.

```
pip install -r requirements
```

## Todos

* Gym interface: Start with assigning only 1 BS (no subset yet)
* Random agent
* Simple RL baseline agent
* Proper wireless model
* (Make decisions in fixed decision interval of XY simulator steps)
* (Extend to subset selection (test different options?))
* As soon as centralized works: Move to distributed RL. Even before adding offloading.
