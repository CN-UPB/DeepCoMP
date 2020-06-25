# deep-rl-mobility-management

Using deep RL for mobility management.

![example](docs/gifs/v04.gif)

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

## Research

### Findings

* Binary observations: [BS available?, BS connected?] work very well
* Replacing binary "BS available?" with achievable data rate by BS does not work at all
* Probably, because data rate is magnitudes larger (up to 150x) than "BS connected?" --> agent becomes blind to 2nd part of obs
* Just cutting the data rate off at some small value (eg, 3 Mbit/s) leads to much better results
* Agent keeps trying to connect to all BS, even if out of range. --> Subtracting req. dr by UE + higher penalty (both!) solves the issue
* Normalizing loses info about which BS has enough dr and connectivity --> does not work as well


### Todos

* Multiple UEs: 
    * Multi-agent: Separate agents for each UE. I should look into ray/rllib: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    * Collaborative learning: Share experience or gradients to train agents together. Use same NN. Later separate NNs? Federated learing.
* Improve radio model: See notes in model.md (fairness, scheduling, freq. reuse, S*c > N)
* Generic utlitiy function: Currently, reward is a step function (pos if enough rate, neg if not). Could also be any other function of the rate, eg, logarithmic
* Efficient caching of connection data rate:
    * Currently always recalculate the data rate per connection per UE, eg, when calculating reward or checking whether we can connect
    * Safe & easy, but probably slow for many UEs/BSs. Let's see
    * Instead, write the dr per connection into a dict (conn --> curr dr); then derive total curr connection etc from that in O(1)
    * Needs to be updated whenever the UE moves or any UE changes its connections (this or another UE)
    * Eg, 1st move all UEs, 2nd check & update connections of all UEs, 3rd calculate reward etc

## Development

### Branches and RL frameworks

* The latest version uses the [RLlib](https://docs.ray.io/en/latest/rllib.html) library for multi-agent RL.
* There is also an older version using [stable_baselines](https://stable-baselines.readthedocs.io/en/master/) for single-agent RL
in the [stable_baselines branch](https://github.com/CN-UPB/deep-rl-mobility-management/tree/stable_baselines) (used for v0.1-v0.3).
* The RLlib version on the `rllib` branch is functionally roughly equivalent to the `stable_baselines` branch (same model, MDP, agent), just with a different framework.
* Development continues in the `dev` branch.
* The current version on `master` and `dev` do not support `stable_baselines` anymore.

### Multi-Agent RL with rllib

* Seems like rllib already supports multi-agent environments
* Anyway seems like the (by far) most complex/feature rich but also mature RL framework
* Doesn't run on Windows yet: https://github.com/ray-project/ray/issues/631 (but should on WSL)
* Multi agent environments: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
* Multi agent concept/policies: https://docs.ray.io/en/latest/rllib-concepts.html#policies-in-multi-agent
* Also supports parameter sharing for joint learning; hierarchical RL etc --> rllib is the way to go
* It's API both for agents and environments (and everything else) is completely different

### Notes on RLlib

#### Training

* `agent.train()` runs one training iteration. Calling it in a loop, continues training for multiple iterations.
* The number of environment steps (not episodes) per iteration is set in `config['train_batch_size']`
* `config['sgd_minibatch_size']` sets how many steps/experiences are used per training epoch
* `config['train_batch_size'] >= config['sgd_minibatch_size']`
* I still don't quite get the details. Sometimes, `config['sgd_minibatch_size']` is ignored and RLlib just trains longer.
* In the results of each training iteration, 
    * `results['hist_stats']['episode_reward']` is a list of the last 100 episode rewards from all training iterations so far. Useful for plotting.
    * `results['info']['num_steps_trained']` shows the total number of training steps, 
    * which is at most `results['info']['num_steps_sampled']`, based on the `train_batch_size`

### Hyperparameter tuning

* Ray's `tune.run()` can also be used directly to tune hyperparameters.
* The resulting `ExperimentAnalysis` object provides the best parameter configuration and path to the saved logs and agent:
https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
