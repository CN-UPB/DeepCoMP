# importing tensorflow before ray leads to an error!
# see https://github.com/ray-project/ray/issues/8993
import tensorflow
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


ray.init()

# ray config
config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 1

# train on cartpole
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
result = trainer.train()
pretty_print(result)
