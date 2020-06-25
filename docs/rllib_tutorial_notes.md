# RLlib tutorial (24.06.2020)

2h tutorial on ray's RLlib via Anyscale Academy: https://anyscale.com/event/rllib-deep-dive/

Held by [Dean Wampler, Head of Developer Relations at Anyscale](https://www.linkedin.com/in/deanwampler/)

Code: https://github.com/anyscale/academy

More events: https://anyscale.com/events/

## My Questions

Questions I had up front:

- How to configure training steps? What knobs to turn? Some settings like batch size are sometimes ignored/overruled? See Readme
- How should I set train_batch_size? any drawback from keeping it small?
- How to get/export/plot training results? How to get the directory name where the training stats and checkpoints are in?
  - No way to do that automatically at the moment
- How to configure or return save path of agent
  - With `analysis = ray.tune.run(checkpoint_at_end=True)`
  - Then `analysis.get_best_checkpoint()` returns the checkpoint
- What's the difference between 0 and 1 worker?

## Notes

* `ray.init()` has useful args:
  * `local_mode`: Set to true to run code sequentially - for debugging!
  * `log_to_driver`: Flase, disables log outputs?
* Useful config option:
  * `config['model']['fcnet_hiddens'] = [20, 20]`  configures the size of the NN
* Ray 0.8.6 was just released with Windows support (alpha version)! https://github.com/ray-project/ray/releases/tag/ray-0.8.6
  * Also support for variable-length observation spaces and arbitrarily nested action spaces.