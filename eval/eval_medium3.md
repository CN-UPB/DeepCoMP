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
* Otherwise, not sure why PPO central is not learning well; Nor how to improve ppo-multi

### Standard obs, normalized rewards, reduced penalty for unsuccessful conn attempts

-1 instead of -3. Similar to lost connections. -3 seemed a bit arbitrary.

* greedy-best
    * Simulation complete            eps_length=50 mean_eps_reward=36.624 mean_eps_time=0.613 mean_step_reward=0.732 num_episodes=100 std_eps_reward=20.172 std_eps_time=0.067
    * exactly as before: never makes unsucc conn attempts anyways
* greedy-all
    * Simulation complete            eps_length=50 mean_eps_reward=43.24 mean_eps_time=0.857 mean_step_reward=0.865 num_episodes=100 std_eps_reward=15.854 std_eps_time=0.145
    * much better because it makes conn attempts all the time and penalty is lower
* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=34.063 mean_eps_time=0.425 mean_step_reward=0.681 num_episodes=100 std_eps_reward=17.254 std_eps_time=0.052
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=43.603 mean_eps_time=0.558 mean_step_reward=0.872 num_episodes=100 std_eps_reward=16.098 std_eps_time=0.076

--> roughly similar results for PPO; greedy-all becomes better

### Same as before but with dr env

Req dr (1, where log func = utility is 0) is subtracted from achievable dr obs --> dr obs in [-1,1] instead of [0,1]
With -1 = can't connect, 0 = 0 utility (req just met), 1 above req

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=39.219 mean_eps_time=0.405 mean_step_reward=0.784 num_episodes=100 std_eps_reward=16.217 std_eps_time=0.051
    * Much better than before +5 avg!
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=43.603 mean_eps_time=0.554 mean_step_reward=0.872 num_episodes=100 std_eps_reward=16.098 std_eps_time=0.074
    * Strange: Exactly the same! --> forgot to change the parent env in multi-agent env (still used same obs as before) --> fix & rerun
    * Simulation complete            eps_length=50 mean_eps_reward=45.775 mean_eps_time=0.542 mean_step_reward=0.916 num_episodes=100 std_eps_reward=16.262 std_eps_time=0.069
    * Also helps here. Better again than greedy-all
    
--> this normalization works quite a bit better! and it should allow working with UEs that have different dr requirements!
  
    
## Adjusted obs: Include distance and velocity

Use normalized rewards from before; And dr env normailzation to -1, 1

### Added distances

Add normalized distance (norm by max distance); no velocity yet

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=39.675 mean_eps_time=0.448 mean_step_reward=0.793 num_episodes=100 std_eps_reward=15.917 std_eps_time=0.057
    * Significantly worse than before. Seems to rather confuse
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=44.885 mean_eps_time=0.549 mean_step_reward=0.898 num_episodes=100 std_eps_reward=16.033 std_eps_time=0.074

### Also added next distances

In addition to curr distance to all BS, also add distance after next step (estimated), ie taking movement direction and velocity into account

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=38.883 mean_eps_time=0.453 mean_step_reward=0.778 num_episodes=100 std_eps_reward=16.127 std_eps_time=0.057
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=44.585 mean_eps_time=0.59 mean_step_reward=0.892 num_episodes=100 std_eps_reward=16.221 std_eps_time=0.072

--> adding distances to obs didn't help (at least within 200k train) and rather hurt
