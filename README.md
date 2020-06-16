# deep-rl-mobility-management

Simple simulation of mobility management scenario to use with deep RL

![example](docs/gifs/v02.gif)

## Setup

Should work with Python 3.6+. Tested with Python 3.7. 
Tensorflow 1 doesn't work on Python 3.8 but is required by stable_baselines.

```
pip install -r requirements
```

For saving gifs, you also need to install [ImageMagick](https://imagemagick.org/index.php).

### Installing RLlib

Ray supports TF2 and thus also Python 3.8.

```
pip install ray[rllib]
```

It may fail installing `gym[atari]`, which needs the following dependencies that can be installed with `apt`:
`cmake, build-essentials, zlib1g-dev`. 
RLlib does not ([yet](https://github.com/ray-project/ray/issues/631)) run on Windows, but it does on WSL.

## Usage

Adjust and run `main.py` in `drl_mobile`:

```
cd drl_mobile
python main.py
```

## Todos

* Fix radio model: See docs
    * Threshold for connecting/disconnecting from a BS
    * Splitting RBs among connected BS rather than achievable dr per user?
* Multiple UEs: 
    * Simple centralized agent that has observations (and actions?) of both UEs combined. Later use as comparison case.
    * Multi-agent: Separate agents for each UE. I should look into ray/rllib: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    * Collaborative learning: Share experience or gradients to train agents together. Use same NN. Later separate NNs? Federated learing.

### Multi-Agent RL with rllib

* Seems like rllib already supports multi-agent environments
* Anyway seems like the (by far) most complex/feature rich but also mature RL framework
* Doesn't run on Windows yet: https://github.com/ray-project/ray/issues/631 (but should on WSL)
* Multi agent environments: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
* Multi agent concept/policies: https://docs.ray.io/en/latest/rllib-concepts.html#policies-in-multi-agent
* Also supports parameter sharing for joint learning; hierarchical RL etc --> rllib is the way to go
* It's API both for agents and environments (and everything else) is completely different

Dev plan:

0. Experiment with rllib and existing environments in separate rl-experiments repo
1. Switch to rllib and verify single-UE case still works as before. Keep working stable baselines code in separate branch
2. Move to multi-user and multi-UE environment with rllib

## Findings

* Binary observations: [BS available?, BS connected?] work very well
* Replacing binary "BS available?" with achievable data rate by BS does not work at all
* Probably, because data rate is magnitudes larger (up to 150x) than "BS connected?" --> agent becomes blind to 2nd part of obs
* Just cutting the data rate off at some small value (eg, 3 Mbit/s) leads to much better results
* Agent keeps trying to connect to all BS, even if out of range. --> Subtracting req. dr by UE + higher penalty (both!) solves the issue
* Normalizing loses info about which BS has enough dr and connectivity --> does not work as well
