
import numpy as np
import gym
from parl.utils import logger
from model import Model
from agent import Agent
from replay_memory import ReplayMemory

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARNING_RATE = 0.001 # 学习率

from parl.algorithms import DQN # 直接从parl库中导入DQN算法，无需自己重写算法
# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def main():
    # 创建环境
    env = gym.make('MountainCar-v0')
    action_dim = env.action_space.n  # MountainCar-v0: 3
    obs_shape = env.observation_space.shape  # MountainCar-v0: (2,)

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池
    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm,obs_dim=obs_shape[0],act_dim=action_dim,e_greed=0.1,  e_greed_decrement=1e-6)
    ckpt = 'model.ckpt'
    agent.restore(ckpt)
    evaluate_reward = evaluate(env,agent,render=True)
if __name__ == '__main__':
    main()