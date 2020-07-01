# deep-rl-mobility-management

Using deep RL for mobility management.

![example](docs/gifs/v06.gif)

## Setup

To install everything, run

```
pip install -r requirements
```

Tested on Ubuntu 20.04 (on WSL) with Python 3.8. RLlib does not ([yet](https://github.com/ray-project/ray/issues/631)) run on Windows, but it does on WSL.

It may fail installing `gym[atari]`, which needs the following dependencies that can be installed with `apt`:
`cmake, build-essentials, zlib1g-dev`. 


For saving videos and gifs, you also need to install ffmpeg (not on Windows) and [ImageMagick](https://imagemagick.org/index.php). 
On Ubuntu:

```
sudo apt install ffmpeg imagemagick
```


## Usage

Adjust and run `main.py` in `drl_mobile`:

```
cd drl_mobile
python main.py
```

Training logs, results, videos, and trained agents are saved in the `training` directory.

#### Tensorboard

To view learning curves (and other metrics) when training an agent, use Tensorboard:

```
tensorboard --logdir training
```

Run the command in a WSL not a PyCharm terminal. Tensorboard is available at http://localhost:6006

## Documentation

* See documents in `docs` folder
* See docstrings in code (TODO: generate read-the-docs in the end for v1.0)

## Research

### Todos

* Nicer visualization: Show total curr dr for each UE in legend; change conn color to green if req dr is satisfied
* Larger scenarios with more UEs and BS. Auto create rand BS, UE; just configure number in env.
* Fix structlog: Once deepcopy is included in release, update requirements to new version & test
* Multiple UEs: 
    * Multi-agent: Separate agents for each UE. I should look into ray/rllib: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    * Collaborative learning: Share experience or gradients to train agents together. Use same NN. Later separate NNs? Federated learing
    * Possibilities: Higher=better
        1. DONE: Use & train exactly same NN for all UEs (still per UE decisions).
        2. Separate NNs for each agent, but share gradient updates or experiences occationally: https://docs.ray.io/en/latest/rllib-env.html#implementing-a-centralized-critic
* Generic utlitiy function: Currently, reward is a step function (pos if enough rate, neg if not). Could also be any other function of the rate, eg, logarithmic
* Efficient caching of connection data rate:
    * Currently always recalculate the data rate per connection per UE, eg, when calculating reward or checking whether we can connect
    * Safe & easy, but probably slow for many UEs/BSs. Let's see
    * Instead, write the dr per connection into a dict (conn --> curr dr); then derive total curr connection etc from that in O(1)
    * Needs to be updated whenever the UE moves or any UE changes its connections (this or another UE)
    * Eg, 1st move all UEs, 2nd check & update connections of all UEs, 3rd calculate reward etc

### Findings

* Binary observations: (BS available?, BS connected?) work very well
* Replacing binary "BS available?" with achievable data rate by BS does not work at all
* Probably, because data rate is magnitudes larger (up to 150x) than "BS connected?" --> agent becomes blind to 2nd part of obs
* Just cutting the data rate off at some small value (eg, 3 Mbit/s) leads to much better results
* Agent keeps trying to connect to all BS, even if out of range. --> Subtracting req. dr by UE + higher penalty (both!) solves the issue
* Normalizing loses info about which BS has enough dr and connectivity --> does not work as well
* Central agent with observations and actions for all UEs in every time step works fine with 2 UEs
* Even with rate-fair sharing, agent tends to connect UEs as long as possible (until connection drops) rather than actively disconnecting UEs that are far away
* This is improved by adding a penalty for losing connections (without active disconnect) and adding obs about the total current dr of each UE (from all connections combined)
* Adding this extra obs about total UE dr (over all BS connections) seems to slightly improve reward, but not a lot
* Multi-agent RL learns better results more quickly than a centralized RL agent
    * Multi-agents using the same NN vs. separate NNs results in comparable performance (slightly worse with separate NN). 
    * Theoretically, separate NNs should take more training as they only see one agent's obs, but allow learning different policies for different agents (eg, slow vs fast UEs)

## Development

* The latest version uses the [RLlib](https://docs.ray.io/en/latest/rllib.html) library for multi-agent RL.
* There is also an older version using [stable_baselines](https://stable-baselines.readthedocs.io/en/master/) for single-agent RL
in the [stable_baselines branch](https://github.com/CN-UPB/deep-rl-mobility-management/tree/stable_baselines) (used for v0.1-v0.3).
* The RLlib version on the `rllib` branch is functionally roughly equivalent to the `stable_baselines` branch (same model, MDP, agent), just with a different framework.
* Development continues in the `dev` branch.
* The current version on `master` and `dev` do not support `stable_baselines` anymore.
