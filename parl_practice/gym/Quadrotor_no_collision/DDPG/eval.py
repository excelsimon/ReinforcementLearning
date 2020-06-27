# coding: utf-8
import numpy as np
from parl.utils import logger
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory  # 经验回放
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from model import QuadrotorModel
from agent import QuadrotorAgent
from parl.algorithms import DDPG

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 2e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent,render=False):
    eval_reward = []
    for i in range(2):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            pred_action = agent.predict(batch_obs.astype('float32'))
            pred_action = np.squeeze(pred_action)
            env_action = pred_action[0] + 0.2 * pred_action[1:]
            env_action = np.clip(env_action, -1.0, 1.0)
            env_action = action_mapping(env_action, env.action_space.low[0], env.action_space.high[0])
            next_obs, reward, done, info = env.step(env_action)
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
    env = make_env("Quadrotor", task="no_collision", seed=1)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] + 1
    model = QuadrotorModel(act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm, obs_dim, act_dim)
    ckpt = 'steps_970464_reward_467.17.ckpt'
    agent.restore(ckpt)
    evaluate_reward = evaluate(env, agent,render=True)
    logger.info('Evaluate reward: {}'.format(evaluate_reward))

if __name__ == '__main__':
    main()