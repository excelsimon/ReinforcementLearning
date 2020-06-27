#-*- coding: utf-8 -*-
from ple.games.snake import Snake
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np
import os
from parl.utils import logger
import os
from utils import get_obs
from agent import Agent
from model import Model
from algorithm import PolicyGradient  # from parl.algorithms import PolicyGradient

from parl.utils import logger

LEARNING_RATE = 1e-3


# 训练一个episode
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    env.reset_game()
    obs = get_obs(env)
    while True:
        obs_list.append(obs)
        action_index = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        action = env.getActionSet()[action_index]
        action_list.append(action_index)

        # 行动
        reward = env.act(action)
        next_obs = get_obs(env)
        done = env.game_over()
        obs = next_obs
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        env.reset_game()
        obs = get_obs(env)
        episode_reward = 0
        while True:
            action_index = agent.predict(obs)  # 选取最优动作
            action = env.getActionSet()[action_index]
            # 行动
            reward = env.act(action)
            next_obs = get_obs(env)
            done = env.game_over()
            obs = next_obs
            episode_reward += reward
            if render:
                env.getScreenRGB()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    # 创建环境
    game = Snake(width=256, height=256, init_length=10)
    p = PLE(game, fps=30, display_screen=True, force_fps=True)
    # 根据parl框架构建agent
    p.reset_game()
    print(p.getActionSet())
    act_dim = len(p.getActionSet())
    obs_dim = 256 * 256

    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # # 加载模型
    # if os.path.exists('model_dir/pg_pong_episode_19.ckpt'):
    #     agent.restore('model_dir/pg_pong_episode_19.ckpt')

    best_total_reward = -float('inf')
    for i in range(50000):
        obs_list, action_list, reward_list = run_episode(p, agent)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)
        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 50 == 0:
            total_reward = evaluate(p, agent, render=True)
            if total_reward>best_total_reward:
                best_total_reward = total_reward
                agent.save('model_dir/pg_pong_episode_{}_reward_{}.ckpt'.format(i,total_reward))
            logger.info('Test reward: {}'.format(total_reward))

    # # save the parameters to ./model.ckpt
    # agent.save('./model.ckpt')


if __name__ == '__main__':
    main()