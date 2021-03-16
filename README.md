[![PyPi](https://github.com/CN-UPB/DeepCoMP/actions/workflows/python-publish.yml/badge.svg?branch=v1.1.0)](https://github.com/CN-UPB/DeepCoMP/actions/workflows/python-publish.yml)
[![DeepSource](https://deepsource.io/gh/CN-UPB/DeepCoMP.svg/?label=active+issues)](https://deepsource.io/gh/CN-UPB/DeepCoMP/?ref=repository-badge)

# DeepCoMP: Self-Learning Dynamic Multi-Cell Selection for Coordinated Multipoint (CoMP)

Deep reinforcement learning for dynamic multi-cell selection in CoMP scenarios.
Three variants: DeepCoMP (central agent), DD-CoMP (distributed agents using central policy), D3-CoMP (distributed agents with separate policies).

![dashboard](https://raw.githubusercontent.com/CN-UPB/DeepCoMP/master/docs/gifs/dashboard.gif?raw=true)
<sup>[Base station icon](https://thenounproject.com/search/?q=base+station&i=1286474) by Clea Doltz from the Noun Project</sup>

## Setup

You need Python 3.8+. You can install `deepcomp` either directly from [PyPi](https://pypi.org/project/deepcomp/) or manually after cloning this repository.

### Simple Installation via PyPi

```
pip install deepcomp
```

### Manual Installation from Source

Clone the repository. Then install everything, following these steps:

```
# only on ubuntu
sudo apt update
sudo apt upgrade
sudo apt install cmake build-essential zlib1g-dev python3-dev

# install all python dependencies
pip install .
# "python setup.py install" does not work for some reason: https://stackoverflow.com/a/66267232/2745116
# for development install (when changing code): pip install -e .
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
deepcomp --env medium --slow-ues 3 --agent central --workers 2 --train-steps 50000 --seed 42 --video both
```

To run DeepCoMP, use `--alg ppo --agent central`.
For DD-CoMP, use `--alg ppo --agent multi`, and for D3-CoMP, use `--alg ppo --agent multi --separate-agent-nns`.

By default, training logs, results, videos, and trained agents are saved in `<project-root>/results`,
where `<project-root>` is the root directory of DeepCoMP.
If you cloned the repo from GitHub, this is where the Readme is. 
If you installed via PyPi, this is in your virtualenv's site packages.
You can choose a custom location with `--result-dir <custom-path>`.

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
tensorboard --logdir results/train/ (--host 0.0.0.0)
```

Tensorboard is available at http://localhost:6006 (or `<remote-ip>:6006` when running remotely).

#### Scaling Up: Running DeepCoMP on multiple cores or a multi-node cluster

To train DeepCoMP on multiple cores in parallel, configure the number of workers (corresponding to CPU cores) with `--workers`.

To scale training to a multi-node cluster, adjust `cluster.yaml` and follow the steps described [here](https://stefanbschneider.github.io/blog/rllib-private-cluster).
Set `--workers` to the total number of CPU cores you want to use on the entire cluster.



## Documentation

API documentation is on [https://cn-upb.github.io/DeepCoMP/](https://cn-upb.github.io/DeepCoMP/).

Documentation is generated based on docstrings using [pdoc3](https://pdoc3.github.io/pdoc/):

```
# from project root
pip install pdoc3
pdoc --force --html --output-dir docs deepcomp
# move files to be picked up by GitHub pages
mv docs/deepcomp/ docs/
# then manually adjust index.html to link to GitHub repo
```

## Contributions

Development: [@stefanbschneider](https://github.com/stefanbschneider/)

Feature requests, questions, issues, and pull requests via GitHub are welcome.

## Acknowledgement

![Huawei logo](https://raw.githubusercontent.com/CN-UPB/DeepCoMP/master/docs/logos/huawei_horizontal.png?raw=true)

[Base station icon](https://thenounproject.com/search/?q=base+station&i=1286474) (used in rendered videos) by Clea Doltz from the Noun Project
