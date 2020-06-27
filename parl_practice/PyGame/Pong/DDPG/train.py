# -*- coding: utf-8 -*-
import os
import numpy as np
from ple.games.pong import Pong
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import os
from parl.utils import logger
from parl.algorithms import DQN
import random
import collections

from parl.utils import logger  # 日志打印工具
import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory  # 经验回放
from utils import get_obs
from pong_model import PongModel
from pong_agent import PongAgent
from parl.algorithms import DDPG

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.001  # Actor网络更新的 learning rate
CRITIC_LR = 0.005  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward
# 训练一个episode

def run_episode(env, agent, rpm):
    total_reward = 0
    env.reset_game()
    obs = get_obs(env)
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        pred_action = agent.predict(batch_obs.astype('float32'))  # 选取最优动作
        pred_action = np.squeeze(pred_action)
        action_set = env.getActionSet()
        action_index = np.random.choice(range(3), p=pred_action)
        action = action_set[action_index]
        reward = env.act(action)
        next_obs = get_obs(env)
        done = env.game_over()
        rpm.append(obs, action_index, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
            batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        env.reset_game()
        obs = get_obs(env)
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            pred_action = agent.predict(batch_obs.astype('float32'))  # 选取最优动作
            pred_action = np.squeeze(pred_action)
            action_set = env.getActionSet()
            action_index = np.random.choice(range(3),p=pred_action)
            action = action_set[action_index]
            reward = env.act(action)
            next_obs = get_obs(env)
            done = env.game_over()

            obs = next_obs
            total_reward += reward
            steps += 1

            if render:
                env.getScreenRGB()
            if done:
                break

        eval_reward.append(total_reward)
    return np.mean(eval_reward)

def main():
    # 创建环境
    game = Pong(width=200, height=200, MAX_SCORE=11)
    p = PLE(game, fps=30, display_screen=True, force_fps=False)
    p.reset_game()
    # 根据parl框架构建agent
    print(p.getActionSet())
    act_dim = len(p.getActionSet())
    print("act_dim:",act_dim)

    obs_dim = 200*200
    # 使用parl框架搭建Agent：QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
    model = PongModel(act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = PongAgent(algorithm, obs_dim, act_dim)
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

    max_episode = 20000
    # 开始训练
    episode = 0
    best_reward = -float('inf')
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward = run_episode(p, agent, rpm)
            episode += 1
        # test part
        eval_reward = evaluate(p, agent, render=True)  # render=True 查看显示效果
        if eval_reward > best_reward:
            best_reward = eval_reward
            agent.save('model_dir/ddpg_pong_{}.ckpt'.format(episode))
        logger.info('episode:{}   test_reward:{}'.format(episode, eval_reward))

if __name__ == '__main__':
    main()