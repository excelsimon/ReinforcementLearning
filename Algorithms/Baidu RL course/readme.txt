1. value based:  Sarsa, Q-learning, DQN
2. policy based: Policy Gradient, DDPG(Deterministic deep policy gradient)
3. on policy: sarsa
4. off policy: Q-learning DQN
5. Sarsa,Q-learning解决状态可数，action可数的情况
   DQN的提出是为了解决state状态不可数，action还是离散的可数的，本质上还是Q-learning
6. Off policy的意思是更新网络用的状态和实际执行的action不一样即target策略和行为策略是不同的