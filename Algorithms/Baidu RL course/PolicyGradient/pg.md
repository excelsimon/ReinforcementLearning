# Lesson 4 策略梯度方法求解RL——Policy Gradient

## 1. Policy Gradient简介
* 在强化学习中，有两大类方法，一种基于值（`Value-based`），一种基于策略（`Policy-based`）
    * `Value-based`的算法的典型代表为`Q-learning`和`SARSA`，将`Q`函数优化到最优，再根据`Q`函数取最优策略。
    * `Policy-based`的算法的典型代表为`Policy Gradient`，直接优化策略函数。
* 采用神经网络拟合策略函数，需计算策略梯度用于优化策略网络。
    * 优化的目标是在策略`π(s,a)`的期望回报：所有的轨迹获得的回报`R`与对应的轨迹发生概率`p`的加权和，当N足够大时，可通过采样N个Episode求平均的方式近似表达。
    
    ![](https://ai-studio-static-online.cdn.bcebos.com/eb184ddf8dcc4dc3b528a105f8d8e3ea6487d4905bc04cdebd7725f2d6a2752f)
    
    * 优化目标对参数`θ`求导后得到策略梯度：
    
    ![](https://ai-studio-static-online.cdn.bcebos.com/326d8abe040347cea25e4c0be3e09015e85cb818a02c445483381540ab1d238c)
    
    
## 2. Policy Gradient实践——REINFORCE算法
* 使用`REINFORCE`解决 连续控制版本的`CartPole`问题，向小车提供推力使得车上的摆杆倒立起来。

### Step1 安装依赖


```python
!pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
!pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

!pip install gym
!pip install paddlepaddle==1.6.3
!pip install parl==1.3.1

# 说明：安装日志中出现两条红色的关于 paddlehub 和 visualdl 的 ERROR 与parl无关，可以忽略，不影响使用
```


```python
# 检查依赖包版本是否正确
!pip list | grep paddlepaddle
!pip list | grep parl
```

### Step2 导入依赖


```python
import os
import gym
import numpy as np

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger

```

### Step3 设置超参数


```python
LEARNING_RATE = 1e-3
```

### Step4 搭建Model、Algorithm、Agent架构
* `Agent`把产生的数据传给`algorithm`，`algorithm`根据`model`的模型结构计算出`Loss`，使用`SGD`或者其他优化器不断的优化，`PARL`这种架构可以很方便的应用在各类深度强化学习问题中。

#### （1）Model
`Model`用来定义前向(`Forward`)网络，用户可以自由的定制自己的网络结构。


```python
class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        out = self.fc1(obs)
        out = self.fc2(out)
        return out

```

#### （2）Algorithm
* `Algorithm` 定义了具体的算法来更新前向网络(`Model`)，也就是通过定义损失函数来更新`Model`，和算法相关的计算都放在`algorithm`中。


```python
# from parl.algorithms import PolicyGradient # 也可以直接从parl库中导入PolicyGradient算法，无需重复写算法

class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        act_prob = self.model(obs)  # 获取输出动作概率
        # log_prob = layers.cross_entropy(act_prob, action) # 交叉熵
        log_prob = layers.reduce_sum(
            -1.0 * layers.log(act_prob) * layers.one_hot(
                action, act_prob.shape[1]),
            dim=1)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)

        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost

```

#### （3）Agent
* `Agent`负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也一般定义在这里。


```python
class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost

```

### Step 5 Training && Test（训练&&测试）


```python
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

```

### Step6 创建环境和Agent，启动训练，保存模型


```python
# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


# 创建环境
env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

# 加载模型
# if os.path.exists('./model.ckpt'):
#     agent.restore('./model.ckpt')
#     run_episode(env, agent, train_or_test='test', render=True)
#     exit()

for i in range(1000):
    obs_list, action_list, reward_list = run_episode(env, agent)
    if i % 10 == 0:
        logger.info("Episode {}, Reward Sum {}.".format(
            i, sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, batch_reward)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(env, agent, render=False) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
        logger.info('Test reward: {}'.format(total_reward))

# 保存模型到文件 ./model.ckpt
agent.save('./model.ckpt')

```

    [32m[06-09 23:30:14 MainThread @<ipython-input-9-67de6cabb2c8>:13][0m obs_dim 4, act_dim 2
    [32m[06-09 23:30:14 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-09 23:30:14 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-09 23:30:14 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 0, Reward Sum 28.0.
    [32m[06-09 23:30:15 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 10, Reward Sum 23.0.
    [32m[06-09 23:30:15 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 20, Reward Sum 68.0.
    [32m[06-09 23:30:16 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 30, Reward Sum 22.0.
    [32m[06-09 23:30:16 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 40, Reward Sum 10.0.
    [32m[06-09 23:30:17 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 50, Reward Sum 18.0.
    [32m[06-09 23:30:17 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 60, Reward Sum 19.0.
    [32m[06-09 23:30:18 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 70, Reward Sum 49.0.
    [32m[06-09 23:30:19 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 80, Reward Sum 76.0.
    [32m[06-09 23:30:20 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 90, Reward Sum 35.0.
    [32m[06-09 23:30:21 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 61.0
    [32m[06-09 23:30:21 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 100, Reward Sum 30.0.
    [32m[06-09 23:30:21 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 110, Reward Sum 42.0.
    [32m[06-09 23:30:22 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 120, Reward Sum 36.0.
    [32m[06-09 23:30:23 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 130, Reward Sum 29.0.
    [32m[06-09 23:30:24 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 140, Reward Sum 38.0.
    [32m[06-09 23:30:24 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 150, Reward Sum 96.0.
    [32m[06-09 23:30:25 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 160, Reward Sum 13.0.
    [32m[06-09 23:30:26 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 170, Reward Sum 51.0.
    [32m[06-09 23:30:27 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 180, Reward Sum 31.0.
    [32m[06-09 23:30:27 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 190, Reward Sum 36.0.
    [32m[06-09 23:30:29 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 51.6
    [32m[06-09 23:30:29 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 200, Reward Sum 29.0.
    [32m[06-09 23:30:30 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 210, Reward Sum 75.0.
    [32m[06-09 23:30:31 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 220, Reward Sum 78.0.
    [32m[06-09 23:30:32 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 230, Reward Sum 35.0.
    [32m[06-09 23:30:32 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 240, Reward Sum 22.0.
    [32m[06-09 23:30:33 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 250, Reward Sum 37.0.
    [32m[06-09 23:30:34 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 260, Reward Sum 105.0.
    [32m[06-09 23:30:35 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 270, Reward Sum 22.0.
    [32m[06-09 23:30:36 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 280, Reward Sum 49.0.
    [32m[06-09 23:30:37 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 290, Reward Sum 116.0.
    [32m[06-09 23:30:39 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 125.8
    [32m[06-09 23:30:39 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 300, Reward Sum 61.0.
    [32m[06-09 23:30:41 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 310, Reward Sum 48.0.
    [32m[06-09 23:30:41 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 320, Reward Sum 50.0.
    [32m[06-09 23:30:43 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 330, Reward Sum 52.0.
    [32m[06-09 23:30:44 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 340, Reward Sum 72.0.
    [32m[06-09 23:30:45 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 350, Reward Sum 107.0.
    [32m[06-09 23:30:46 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 360, Reward Sum 41.0.
    [32m[06-09 23:30:48 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 370, Reward Sum 91.0.
    [32m[06-09 23:30:49 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 380, Reward Sum 64.0.
    [32m[06-09 23:30:50 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 390, Reward Sum 87.0.
    [32m[06-09 23:30:53 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 172.2
    [32m[06-09 23:30:53 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 400, Reward Sum 57.0.
    [32m[06-09 23:30:54 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 410, Reward Sum 59.0.
    [32m[06-09 23:30:55 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 420, Reward Sum 56.0.
    [32m[06-09 23:30:57 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 430, Reward Sum 56.0.
    [32m[06-09 23:30:58 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 440, Reward Sum 77.0.
    [32m[06-09 23:30:59 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 450, Reward Sum 30.0.
    [32m[06-09 23:31:01 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 460, Reward Sum 68.0.
    [32m[06-09 23:31:02 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 470, Reward Sum 131.0.
    [32m[06-09 23:31:04 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 480, Reward Sum 75.0.
    [32m[06-09 23:31:06 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 490, Reward Sum 107.0.
    [32m[06-09 23:31:09 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 200.0
    [32m[06-09 23:31:09 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 500, Reward Sum 145.0.
    [32m[06-09 23:31:12 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 510, Reward Sum 119.0.
    [32m[06-09 23:31:13 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 520, Reward Sum 18.0.
    [32m[06-09 23:31:16 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 530, Reward Sum 92.0.
    [32m[06-09 23:31:19 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 540, Reward Sum 140.0.
    [32m[06-09 23:31:21 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 550, Reward Sum 194.0.
    [32m[06-09 23:31:24 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 560, Reward Sum 34.0.
    [32m[06-09 23:31:26 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 570, Reward Sum 67.0.
    [32m[06-09 23:31:29 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 580, Reward Sum 56.0.
    [32m[06-09 23:31:31 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 590, Reward Sum 180.0.
    [32m[06-09 23:31:36 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 200.0
    [32m[06-09 23:31:37 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 600, Reward Sum 200.0.
    [32m[06-09 23:31:39 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 610, Reward Sum 104.0.
    [32m[06-09 23:31:42 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 620, Reward Sum 21.0.
    [32m[06-09 23:31:45 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 630, Reward Sum 200.0.
    [32m[06-09 23:31:48 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 640, Reward Sum 200.0.
    [32m[06-09 23:31:51 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 650, Reward Sum 139.0.
    [32m[06-09 23:31:54 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 660, Reward Sum 200.0.
    [32m[06-09 23:31:57 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 670, Reward Sum 200.0.
    [32m[06-09 23:32:01 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 680, Reward Sum 200.0.
    [32m[06-09 23:32:04 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 690, Reward Sum 200.0.
    [32m[06-09 23:32:09 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 200.0
    [32m[06-09 23:32:09 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 700, Reward Sum 141.0.
    [32m[06-09 23:32:12 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 710, Reward Sum 139.0.
    [32m[06-09 23:32:15 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 720, Reward Sum 140.0.
    [32m[06-09 23:32:18 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 730, Reward Sum 130.0.
    [32m[06-09 23:32:21 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 740, Reward Sum 160.0.
    [32m[06-09 23:32:24 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 750, Reward Sum 25.0.
    [32m[06-09 23:32:28 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 760, Reward Sum 170.0.
    [32m[06-09 23:32:32 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 770, Reward Sum 200.0.
    [32m[06-09 23:32:36 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 780, Reward Sum 200.0.
    [32m[06-09 23:32:39 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 790, Reward Sum 103.0.
    [32m[06-09 23:32:45 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 200.0
    [32m[06-09 23:32:45 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 800, Reward Sum 185.0.
    [32m[06-09 23:32:49 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 810, Reward Sum 200.0.
    [32m[06-09 23:32:52 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 820, Reward Sum 95.0.
    [32m[06-09 23:32:55 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 830, Reward Sum 98.0.
    [32m[06-09 23:32:59 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 840, Reward Sum 200.0.
    [32m[06-09 23:33:03 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 850, Reward Sum 200.0.
    [32m[06-09 23:33:07 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 860, Reward Sum 166.0.
    [32m[06-09 23:33:11 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 870, Reward Sum 200.0.
    [32m[06-09 23:33:14 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 880, Reward Sum 163.0.
    [32m[06-09 23:33:18 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 890, Reward Sum 200.0.
    [32m[06-09 23:33:23 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 200.0
    [32m[06-09 23:33:24 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 900, Reward Sum 165.0.
    [32m[06-09 23:33:28 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 910, Reward Sum 200.0.
    [32m[06-09 23:33:31 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 920, Reward Sum 200.0.
    [32m[06-09 23:33:34 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 930, Reward Sum 38.0.
    [32m[06-09 23:33:38 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 940, Reward Sum 146.0.
    [32m[06-09 23:33:41 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 950, Reward Sum 192.0.
    [32m[06-09 23:33:45 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 960, Reward Sum 152.0.
    [32m[06-09 23:33:48 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 970, Reward Sum 200.0.
    [32m[06-09 23:33:52 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 980, Reward Sum 200.0.
    [32m[06-09 23:33:56 MainThread @<ipython-input-9-67de6cabb2c8>:30][0m Episode 990, Reward Sum 160.0.
    [32m[06-09 23:34:01 MainThread @<ipython-input-9-67de6cabb2c8>:39][0m Test reward: 200.0

