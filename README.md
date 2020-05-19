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

## Usage

Adjust and run `main.py` in `drl_mobile`:

```
cd drl_mobile
python main.py
```

## Todos

* Multiple UEs: 
    * Multi-agent: Separate agents for each UE. I should look into ray/rllib: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    * Collaborative learning: Share experience or gradients to train agents together. Use same NN. Later separate NNs? Federated learing.

## Findings

* Binary observations: [BS available?, BS connected?] work very well
* Replacing binary "BS available?" with achievable data rate by BS does not work at all
* Probably, because data rate is magnitudes larger (up to 150x) than "BS connected?" --> agent becomes blind to 2nd part of obs
* Just cutting the data rate off at some small value (eg, 3 Mbit/s) leads to much better results
* Agent keeps trying to connect to all BS, even if out of range. --> Subtracting req. dr by UE + higher penalty (both!) solves the issue
* Normalizing loses info about which BS has enough dr and connectivity --> does not work as well
