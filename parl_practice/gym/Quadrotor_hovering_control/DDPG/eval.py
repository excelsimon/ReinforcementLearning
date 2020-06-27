#!/usr/bin/env python
# coding: utf-8
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境

ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward

from parl.algorithms import DDPG
from agent import QuadrotorAgent
from model import QuadrotorModel

import numpy as np
def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = np.clip(np.random.normal(action, 1), -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])
        new_action = [0]*(action.shape[0]-1)
        for i in range(len(new_action)):
            new_action[i] = action[0]+0.3*action[i+1]
        new_action = np.array(new_action)
        next_obs, reward, done, info = env.step(new_action)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs,                     batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)
        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent,render=True):
    eval_reward = []
    for i in range(1):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action = np.clip(action, -1.0, 1.0)
            action = action_mapping(action, env.action_space.low[0], 
                                    env.action_space.high[0])
            new_action = [0] * (action.shape[0] - 1)
            for i in range(len(new_action)):
                new_action[i] = action[0] + 0.3 * action[i + 1]
            new_action = np.array(new_action)
            #new_action = action[0] + 0.3*action[1:]
            next_obs, reward, done, info = env.step(new_action)

            obs = next_obs
            total_reward += reward
            steps += 1
            if render:
                env.render()
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

def main():
    # 创建飞行器环境
    env = make_env("Quadrotor_hovering_control", task="hovering_control")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(obs_dim,act_dim)

    model = QuadrotorModel(act_dim+1)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm, obs_dim, act_dim+1)
    ckpt = 'steps_700176.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
    agent.restore(ckpt)
    evaluate_reward = evaluate(env, agent)
    logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward

if __name__ == '__main__':
    main()


