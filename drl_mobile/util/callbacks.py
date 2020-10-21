from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker


class CustomMetricCallbacks(DefaultCallbacks):
    """
    Callbacks for including custom scalar metrics for monitoring with tensorboard
    https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
    """
    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.custom_metrics['test_metric'] = 42
        # get the info dict for the first UE (it's the same for all)
        ue_id = base_env.envs[0].ue_list[0].id
        info = episode.last_info_for(ue_id)
        # add all custom scalar metrics
        if 'scalar_metrics' in info:
            for metric_name, metric_value in info['scalar_metrics'].items():
                episode.custom_metrics[metric_name] = metric_value
