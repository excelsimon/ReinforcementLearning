#-*- coding: utf-8 -*-

from ple.games.snake import Snake
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np
import os
from parl.utils import logger
from parl.algorithms import DQN
import random
import collections

from parl.utils import logger  # 日志打印工具

from model import Model
from algorithm import DQN  # from parl.algorithms import DQN  # parl >= 1.3.1
from agent import Agent
from replay_memory import ReplayMemory
from utils import get_obs

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(10):
        env.reset_game()
        obs = get_obs(env)
        episode_reward = 0
        while True:
            action_index = agent.predict(obs)  # 选取最优动作
            action = env.getActionSet()[action_index]
            reward = env.act(action)
            obs = get_obs(env)
            episode_reward += reward
            if render:
                env.getScreenRGB()
            if env.game_over():
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    # 创建环境
    game = Snake(width=224, height=224, init_length=7)
    p = PLE(game, fps=30, display_screen=True, force_fps=True)
    # 根据parl框架构建agent
    print(p.getActionSet())
    act_dim = len(p.getActionSet())

    #rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    model = Model(act_dim=act_dim)
    alg = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(alg, act_dim=act_dim, e_greed_decrement=1e-6, e_greed=0.2)  # e_greed有一定概率随机选取动作，探索

    # 加载模型
    if os.path.exists('./dqn_snake_7.ckpt'):
        agent.restore('./dqn_snake_7.ckpt')
    evaluate(p,agent,False)

if __name__ == '__main__':
    main()