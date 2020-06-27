import gym
env = gym.make("BipedalWalkerHardcore-v3")
obs_space = env.observation_space
act_space = env.action_space
print("obs_space:",obs_space)
print("act_space:",act_space)
print("high:",env.observation_space.high)
print("low:",env.observation_space.low)
for i_episode in range(20):
    obs = env.reset()
    for t in range(100):
        env.render()
        print("obs:",obs)
        action = env.action_space.sample()
        print("action:",action)
        obs, reward, done, info = env.step(action)
        print("reward:",reward)
        print("done:",done)
        print("info:",info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()