# DeepCoMP: Self-Learning Dynamic Multi-Cell Selection for Coordinated Multipoint (CoMP)

Deep reinforcement learning for dynamic multi-cell selection in CoMP scenarios.
Three variants: DeepCoMP (central agent), DD-CoMP (distributed agents using central policy), D3-CoMP (distributed agents with separate policies).

**[API documentation](deepcomp/index.html)**

![example](gifs/v10.gif)


## Setup

You need Python 3.8+.
To install everything, run

```
# only on ubuntu
sudo apt update
sudo apt upgrade
sudo apt install cmake build-essential zlib1g-dev python3-dev

# then install rllib and structlog manually for now
pip install ray[rllib]==1
pip install git+https://github.com/stefanbschneider/structlog.git@dev

# complete installation of remaining dependencies
python setup.py install
```

Tested on Ubuntu 20.04 and Windows 10 with Python 3.8.

For saving videos and gifs, you also need to install ffmpeg (not on Windows) and [ImageMagick](https://imagemagick.org/index.php). 
On Ubuntu:

```
sudo apt install ffmpeg imagemagick
```


## Usage

```
# get an overview of all options
deepcomp -h
```

For example: 

```
deepcomp --env medium --slow-ues 3 --fast-ues 0 --agent central --workers 2 --train-steps 50000 --seed 42 --video both --sharing mixed
```

To run DeepCoMP, use `--alg ppo --agent central`.
For DD-CoMP, use `--alg ppo --agent multi`, and for D3-CoMP, use `--alg ppo --agent multi --separate-agent-nns`.

Training logs, results, videos, and trained agents are saved in the `results` directory.

#### Accessing results remotely

When running remotely, you can serve the replay video by running:

```
cd results
python -m http.server
```

Then access at `<remote-ip>:8000`.

#### Tensorboard

To view learning curves (and other metrics) when training an agent, use Tensorboard:

```
tensorboard --logdir results/PPO/ (--host 0.0.0.0)
```

Tensorboard is available at http://localhost:6006

