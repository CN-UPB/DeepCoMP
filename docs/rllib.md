# Notes on RLlib

These notes are referring to `ray[rllib]==0.8.6`.

## Multi-Agent RL with rllib

* Seems like rllib already supports multi-agent environments
* Anyway seems like the (by far) most complex/feature rich but also mature RL framework
* Doesn't run on Windows yet: https://github.com/ray-project/ray/issues/631 (but should on WSL)
* Multi agent environments: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
* Multi agent concept/policies: https://docs.ray.io/en/latest/rllib-concepts.html#policies-in-multi-agent
* Also supports parameter sharing for joint learning; hierarchical RL etc --> rllib is the way to go
* It's API both for agents and environments (and everything else) is completely different

## Environment Requirements

* Envs need to follow the Gym interface
* The constructor must take a `env_config` dict as only argument
* The environment and all involved classes need to support `deepcopy`
    * This lead to hard-to-debug errors when I had cyclic references inside my env that did not get copied correctly
    * Best approach: Avoid cyclic references
    * Alternative: Overwrite `deepcopy`

## Training

* `agent.train()` runs one training iteration. Calling it in a loop, continues training for multiple iterations.
* The number of environment steps (not episodes) per iteration is set in `config['train_batch_size']`
* `config['sgd_minibatch_size']` sets how many steps/experiences are used per training epoch
* `config['train_batch_size'] >= config['sgd_minibatch_size']`
* I still don't quite get the details. Sometimes, `config['sgd_minibatch_size']` is ignored and RLlib just trains longer.
* In the results of each training iteration, 
    * `results['hist_stats']['episode_reward']` is a list of the last 100 episode rewards from all training iterations so far. Useful for plotting.
    * `results['info']['num_steps_trained']` shows the total number of training steps, 
    * which is at most `results['info']['num_steps_sampled']`, based on the `train_batch_size`

## Hyperparameter tuning

* Ray's `tune.run()` can also be used directly to tune hyperparameters.
* The resulting `ExperimentAnalysis` object provides the best parameter configuration and path to the saved logs and agent:
https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis



## RLlib tutorial (24.06.2020)

2h tutorial on ray's RLlib via Anyscale Academy: https://anyscale.com/event/rllib-deep-dive/

Held by [Dean Wampler, Head of Developer Relations at Anyscale](https://www.linkedin.com/in/deanwampler/)

Code: https://github.com/anyscale/academy

More events: https://anyscale.com/events/

### My Questions

Questions I had up front:

- How to configure training steps? What knobs to turn? Some settings like batch size are sometimes ignored/overruled? See Readme
- How should I set train_batch_size? any drawback from keeping it small?
- How to get/export/plot training results? How to get the directory name where the training stats and checkpoints are in?
  - No way to do that automatically at the moment
- How to configure or return save path of agent
  - With `analysis = ray.tune.run(checkpoint_at_end=True)`
  - Then `analysis.get_best_checkpoint()` returns the checkpoint --> Tested & doesn't work.
    - Instead `analysis.get_best_logdir(metric='episode_reward_mean')` works
    - `analysis.get_trial_checkpoints_paths(analysis.get_best_trial('episode_reward_mean'), 'episode_reward_mean')` gets me the path to the checkpoint
- What's the difference between 0 and 1 worker?

### Notes

* `ray.init()` has useful args:
  * `local_mode`: Set to true to run code sequentially - for debugging!
  * `log_to_driver`: Flase, disables log outputs?
* Useful config option:
  * `config['model']['fcnet_hiddens'] = [20, 20]`  configures the size of the NN
* Ray 0.8.6 was just released with Windows support (alpha version)! https://github.com/ray-project/ray/releases/tag/ray-0.8.6
  * Also support for variable-length observation spaces and arbitrarily nested action spaces.