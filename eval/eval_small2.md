# Small env, 2 UEs, Proportional-fair, eps len 50, eval 50, seed 42, train for 100k

20.07.2020

## Normal obs: Normlaized dr, connected BS

* greedy-best
    * Simulation complete            eps_length=50 mean_eps_reward=824.009 mean_eps_time=0.39 mean_step_reward=16.48 num_episodes=50 std_eps_reward=309.604 std_eps_time=0.154
* greedy-all
    * Simulation complete            eps_length=50 mean_eps_reward=893.441 mean_eps_time=0.41 mean_step_reward=17.869 num_episodes=50 std_eps_reward=283.857 std_eps_time=0.07
* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=837.3 mean_eps_time=0.309 mean_step_reward=16.746 num_episodes=50 std_eps_reward=273.123 std_eps_time=0.042
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=874.535 mean_eps_time=0.352 mean_step_reward=17.491 num_episodes=50 std_eps_reward=281.041 std_eps_time=0.066

## Extra obs: num UEs per BS (normalized)

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=801.935 mean_eps_time=0.31 mean_step_reward=16.039 num_episodes=50 std_eps_reward=264.537 std_eps_time=0.044
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=856.978 mean_eps_time=0.359 mean_step_reward=17.14 num_episodes=50 std_eps_reward=307.665 std_eps_time=0.057

--> Does not seem to help. If anything makes things a bit worse.


## Normal obs + total dr (clipped & normalized)

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=828.992 mean_eps_time=0.302 mean_step_reward=16.58 num_episodes=50 std_eps_reward=295.524 std_eps_time=0.042
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=859.985 mean_eps_time=0.365 mean_step_reward=17.2 num_episodes=50 std_eps_reward=284.066 std_eps_time=0.058

--> Does not seem to help. If anything makes things a bit worse.


## Num UEs per BS + Total Dr

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=836.458 mean_eps_time=0.309 mean_step_reward=16.729 num_episodes=50 std_eps_reward=277.218 std_eps_time=0.052
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=864.314 mean_eps_time=0.36 mean_step_reward=17.286 num_episodes=50 std_eps_reward=293.184 std_eps_time=0.054

--> comparable to original reward

Again, same thing but trained for 200k instead of 100k:

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=846.191 mean_eps_time=0.305 mean_step_reward=16.924 num_episodes=50 std_eps_reward=294.629 std_eps_time=0.047
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=872.519 mean_eps_time=0.362 mean_step_reward=17.45 num_episodes=50 std_eps_reward=283.521 std_eps_time=0.053

--> slight improvement over 100k, but still comparable to original rewards; seems like agent converged (more training only helps a bit) and additional obs don't help (rather hurt)

## Normalized reward

To -1, 1

### Obs as before (incl ues_at_bs and dr_total)

* greedy-best
    * Simulation complete            eps_length=50 mean_eps_reward=41.2 mean_eps_time=0.365 mean_step_reward=0.824 num_episodes=50 std_eps_reward=15.48 std_eps_time=0.127
* greedy-all
    * Simulation complete            eps_length=50 mean_eps_reward=44.672 mean_eps_time=0.548 mean_step_reward=0.893 num_episodes=50 std_eps_reward=14.193 std_eps_time=0.179
* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=40.508 mean_eps_time=0.312 mean_step_reward=0.81 num_episodes=50 std_eps_reward=13.911 std_eps_time=0.044
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=42.219 mean_eps_time=0.359 mean_step_reward=0.844 num_episodes=50 std_eps_reward=14.012 std_eps_time=0.047
    
    
### Same again without extra obs

* ppo-central
    * Simulation complete            eps_length=50 mean_eps_reward=40.749 mean_eps_time=0.305 mean_step_reward=0.815 num_episodes=50 std_eps_reward=14.436 std_eps_time=0.042
* ppo-multi
    * Simulation complete            eps_length=50 mean_eps_reward=42.347 mean_eps_time=0.358 mean_step_reward=0.847 num_episodes=50 std_eps_reward=13.399 std_eps_time=0.058

--> Reward normalization doesn't seem to make a big difference. Slightly negative if anything. Still better without extra obs (again very slight difference).
