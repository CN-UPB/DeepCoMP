# Eval on medium env with 3 UEs

3 slow UEs, prop-fair sharing, seed 42, eps len 50, eval 100, train 200k steps

High dependence on random seed! Even for heuristics! Because eval episodes are so different!

### Standard obs, normalized rewads

Obs: Norm dr, connected?; Reward: Norm to -1, 1

* greedy-best
    * Simulation complete            eps_length=50 mean_eps_reward=36.624 mean_eps_time=0.614 mean_step_reward=0.732 num_episodes=100 std_eps_reward=20.172 std_eps_time=0.207
* greedy-all
    * Simulation complete            eps_length=50 mean_eps_reward=36.895 mean_eps_time=0.815 mean_step_reward=0.738 num_episodes=100 std_eps_reward=15.579 std_eps_time=0.241
* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=29.516 mean_eps_time=0.427 mean_step_reward=0.59 num_episodes=100 std_eps_reward=17.285 std_eps_time=0.057
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=40.359 mean_eps_time=0.552 mean_step_reward=0.807 num_episodes=100 std_eps_reward=15.077 std_eps_time=0.065

Analysis:
* ppo-central randomly disconnects UEs sometimes

