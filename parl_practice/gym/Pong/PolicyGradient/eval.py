#-*- coding: utf-8 -*-

import os
import gym
import numpy as np
import parl

from agent import Agent
from model import Model
from parl.algorithms import PolicyGradient
from parl.utils import logger
LEARNING_RATE = 1e-3

# Pong 图片预处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2,0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()

def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs)  # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def main():
    # 创建环境
    env = gym.make('Pong-v0')
    obs_dim = 80 * 80
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    if os.path.exists('model.ckpt'):
        agent.restore('model.ckpt')
        print("restore_succeed")

    eval_reward = evaluate(env,agent,render=True)
    return eval_reward

if __name__ == '__main__':
    reward = main()
    print(reward)