# example using rllib: https://docs.ray.io/en/latest/rllib.html#running-rllib
# https://docs.ray.io/en/latest/rllib-training.html#basic-python-api
# since rllib doesn't run on Windows, I set up PyCharm to use WSL: https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.
# 20 training iterations (!= episodes) should be enough for optimal reward
for i in range(10):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Trainer's Policy's ModelV2
# (tf or torch) by doing:
# trainer.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch trainers/models.
