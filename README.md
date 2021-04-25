[![CI](https://github.com/CN-UPB/DeepCoMP/actions/workflows/python-test.yml/badge.svg)](https://github.com/CN-UPB/DeepCoMP/actions/workflows/python-test.yml)
[![PyPi](https://github.com/CN-UPB/DeepCoMP/actions/workflows/python-publish.yml/badge.svg?branch=v1.1.0)](https://github.com/CN-UPB/DeepCoMP/actions/workflows/python-publish.yml)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/stefanbschneider/deepcomp?label=Docker%20Build&logo=docker)](https://hub.docker.com/r/stefanbschneider/deepcomp)
[![Docker Pulls](https://img.shields.io/docker/pulls/stefanbschneider/deepcomp?label=Docker%20Pulls&logo=docker)](https://hub.docker.com/r/stefanbschneider/deepcomp)
[![DeepSource](https://deepsource.io/gh/CN-UPB/DeepCoMP.svg/?label=active+issues)](https://deepsource.io/gh/CN-UPB/DeepCoMP/?ref=repository-badge)


# DeepCoMP: Self-Learning Dynamic Multi-Cell Selection for Coordinated Multipoint (CoMP)

Deep reinforcement learning for dynamic multi-cell selection in CoMP scenarios.
Three variants: DeepCoMP (central agent), DD-CoMP (distributed agents using central policy), D3-CoMP (distributed agents with separate policies).
All three approaches self-learn and adapt to various scenarios in mobile networks without expert knowledge, human intervention, or detailed assumptions about the underlying system.
Compared to other approaches, they are more flexible and achieve higher Quality of Experience.

<p align="center">
    <img src="https://raw.githubusercontent.com/CN-UPB/DeepCoMP/master/docs/gifs/dashboard_lossy.gif?raw=true"><br/>
    <em>Visualized cell selection policy of DeepCoMP after 2M training steps.</em><br>
    <sup><a href="https://thenounproject.com/search/?q=base+station&i=1286474" target="_blank">Base station icon</a> by Clea Doltz from the Noun Project</sup>
</p>

## Setup

You need Python 3.8+. You can install `deepcomp` either directly from [PyPi](https://pypi.org/project/deepcomp/) or manually after cloning this repository.

### Simple Installation via PyPi

```
sudo apt update
sudo apt upgrade
sudo apt install cmake build-essential zlib1g-dev python3-dev

pip install deepcomp
```

### Manual Installation from Source

For adjusting or further developing DeepCoMP, it's better to install manually rather than from PyPi. 
Clone the repository. Then install everything, following these steps:

```
# only on ubuntu
sudo apt update
sudo apt upgrade
sudo apt install cmake build-essential zlib1g-dev python3-dev

# clone
git clone git@github.com:CN-UPB/DeepCoMP.git
cd DeepCoMP

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

### Docker

There is a Docker image that comes with `deepcomp` preinstalled. 
To use the Docker image, simply pull the latest version from [Docker Hub](https://hub.docker.com/r/stefanbschneider/deepcomp):

```
docker pull stefanbschneider/deepcomp
# tag image with just "deepcomp". alternatively, write out "stefanbschneider/deepcomp" in all following commands.
docker tag stefanbschneider/deepcomp:latest deepcomp
```

Alternatively, to build the Docker image manually from the `Dockerfile`, clone this repository and run
```
docker build -t deepcomp .
```
Use the `--no-cache` option is to force a rebuild of the image, pulling the latest `deepcomp` version from PyPI.


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

### Docker

**Note:** By default, results within the Docker container are not stored persistently. 
To save them, copy them from the Docker container or use a Docker volume.

#### Start the Container

If you want to use the `deepcomp` Docker container and pulled the corresponding image from Docker Hub,
you can use it as follows:
```
docker run -d -p 6006:6006 -p 8000:8000 --rm --shm-size=3gb --name deepcomp deepcomp
```
This starts the Docker container in the background, publishing port 6006 for TensorBoard and port 8000 for the
HTTP server (described below).
The container automatically starts TensorBoard and the HTTP server, so this does not need to be done manually.
The `--rm` flag automatically removes the container once it is stopped.
The `--shm-size=3gb` sets the size of `/dev/shm` inside the Docker container to 3 GB, which is too small by default.

#### Use DeepCoMP on the Container

To execute commands on the running Docker container, use `docker exec <container-name> <command>` as follows:
```
docker exec deepcomp deepcomp <deepcomp-args>
```
Here, the arguments are identical with the ones described above.
For example, the following command lists all CLI options:
```
docker exec deepcomp deepcomp -h
```
Or to train the central DeepCoMP agent for a short duration of 4000 steps:
```
docker exec -t deepcomp deepcomp --approach deepcomp --train-steps 4000 --batch-size 200 --ues 2 --result-dir results
```
**Important:** Specify `--result-dir results` as argument. 
Otherwise, the results will be stored elsewhere and TensorFlow and the HTTP server will not find and display them.

The other `deepcomp` arguments can be set as desired.
The Docker `-t` flag ensures that the output is printed continuously during training, not just after completion.

To inspect training progress or view create files (e.g., rendered videos), use TensorBoard and the HTTP server,
which are available via `localhost:6006` and `localhost:8000`.

#### Terminate the Container

**Important:** Stopping the container will remove any files and training progress within the container.

Stop the container with
```
docker stop deepcomp
```

### Accessing results remotely

When running remotely, you can serve the replay video by running:

```
cd results
python -m http.server
```

Then access at `<remote-ip>:8000`.

### Tensorboard

To view learning curves (and other metrics) when training an agent, use Tensorboard:

```
tensorboard --logdir results/train/ (--host 0.0.0.0)
```

Tensorboard is available at http://localhost:6006 (or `<remote-ip>:6006` when running remotely).

### Scaling Up: Running DeepCoMP on multiple cores or a multi-node cluster

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

DeepCoMP is an outcome of a joint project between Paderborn University, Germany, and Huawei Germany.

<p align="center">
    <img src="https://raw.githubusercontent.com/CN-UPB/DeepCoMP/master/docs/logos/upb.png?raw=true" width="200" hspace="30"/>
    <img src="https://raw.githubusercontent.com/CN-UPB/DeepCoMP/master/docs/logos/huawei_horizontal.png?raw=true" width="250" hspace="30"/>
</p>

[Base station icon](https://thenounproject.com/search/?q=base+station&i=1286474) (used in rendered videos) by Clea Doltz from the Noun Project.
