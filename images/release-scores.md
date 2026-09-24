# Release benchmark — final return and wall-clock

Final return is the mean of the last 10 evaluations. The budget is 500M environment steps; an arm that stopped short of it is marked with the steps it reached.

## envpool_cpu

| agent | HumanoidRun return | HumanoidStand return | HumanoidWalk return | HumanoidRun hours | HumanoidStand hours | HumanoidWalk hours |
|---|---|---|---|---|---|---|
| DDPG | 498 | 924 | 812 | 6.8 | 7.4 | 7.0 |
| TD3 | 534 | 922 | 820 | 7.4 | 7.9 | 7.4 |
| D4PG | 692 | 960 | 952 | 10.0 | 9.9 | 10.1 |
| TD4 | 616 | 957 | 908 | 13.4 | 13.5 | 12.7 |
| SAC | 475 | 917 | 841 | 8.8 | 9.5 | 9.3 |
| MPO | 589 | 836 | 775 | 22.1 | 23.1 | 22.8 |
| PPO | 761 | 6 | 830 | 4.8 | 4.8 | 4.6 |

## mjbatch_cpu

| agent | HumanoidRun return | HumanoidStand return | HumanoidWalk return | HumanoidRun hours | HumanoidStand hours | HumanoidWalk hours |
|---|---|---|---|---|---|---|
| PPO | 462 @184M | — | 863 @222M | 2.7 | — | 2.7 |

## warp_gpu

| agent | HumanoidRun return | HumanoidStand return | HumanoidWalk return | HumanoidRun hours | HumanoidStand hours | HumanoidWalk hours |
|---|---|---|---|---|---|---|
| DDPG | 119 | 848 | 846 | 2.3 | 2.2 | 2.5 |
| TD3 | 140 | 860 | 830 | 2.4 | 2.4 | 2.7 |
| D4PG | 161 | 898 | 378 | 2.5 | 2.4 | 2.4 |
| TD4 | 150 | 38 | 985 | 2.6 | 2.5 | 2.8 |
| SAC | 450 | 925 | 573 | 2.6 | 2.5 | 2.7 |
| MPO | 153 | 725 | 826 | 3.0 | 2.9 | 3.0 |
| PPO | 7 | 229 | 25 | 1.8 | 1.8 | 1.9 |
