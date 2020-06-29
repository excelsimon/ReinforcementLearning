# -*- coding: utf-8 -*-

import os
import numpy as np
from parl.utils import logger
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory  # 经验回放
from model import BipedalWalkerModel
from agent import BipedalWalkerAgent
from parl.algorithms import DDPG
import gym
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 10e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent,render):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps+=1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        next_obs, reward, done, info = env.step(action)
        if render:
            env.render()
        obs = next_obs
        total_reward += reward
        if done:
            break
    return total_reward,steps

def main():
    # 创建BipedalWalker环境
    env = gym.make("BipedalWalker-v3")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 使用parl框架搭建Agent：QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
    model = BipedalWalkerModel(act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = BipedalWalkerAgent(algorithm, obs_dim, act_dim)
    ckpt = 'model.ckpt'
    agent.restore(ckpt)
    print("restore succeed")
    return
    best_reward = -float('inf')
    for i in range(5):
        evaluate_reward, steps = evaluate(env, agent,render=False)
        if evaluate_reward>best_reward:
            best_reward = evaluate_reward
        logger.info('Episode:{}, Evaluate reward: {}'.format(i, evaluate_reward))
    print("best_reward:",best_reward)



if __name__ == '__main__':
    main()

