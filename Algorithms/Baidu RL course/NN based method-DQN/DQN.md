# Lesson 3 神经网络方法求解RL——DQN
## 1. DQN简介
* 上节课介绍的表格型方法存储的状态数量有限，当面对围棋或机器人控制这类有数不清的状态的环境时，表格型方法在存储和查找效率上都受局限，`DQN`的提出解决了这一局限，使用神经网络来近似替代`Q`表格。
* 本质上`DQN`还是一个`Q-learning`算法，更新方式一致。为了更好的探索环境，同样的也采用`ε-greedy`方法训练。
* 在`Q-learning`的基础上，`DQN`提出了两个技巧使得`Q`网络的更新迭代更稳定。
    * 经验回放 `Experience Replay`：主要解决样本关联性和利用效率的问题。使用一个经验池存储多条经验`s,a,r,s'`，再从中随机抽取一批数据送去训练。
    * 固定Q目标 `Fixed-Q-Target`：主要解决算法训练不稳定的问题。复制一个和原来`Q`网络结构一样的`Target Q`网络，用于计算`Q`目标值。

## 2. DQN实践
* 使用`DQN`解决CartPole问题，移动小车使得车上的摆杆倒立起来。

### Step1 安装依赖


```python
!pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
!pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

!pip install gym
!pip install paddlepaddle==1.6.3
!pip install parl==1.3.1

# 说明：安装日志中出现两条红色的关于 paddlehub 和 visualdl 的 ERROR 与parl无关，可以忽略，不影响使用
```

### Step2  导入依赖


```python
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
```

### Step3 设置超参数


```python
LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
```

### Step4 搭建Model、Algorithm、Agent架构
* `Agent`把产生的数据传给`algorithm`，`algorithm`根据`model`的模型结构计算出`Loss`，使用`SGD`或者其他优化器不断的优化，`PARL`这种架构可以很方便的应用在各类深度强化学习问题中。

#### （1）Model
* `Model`用来定义前向(`Forward`)网络，用户可以自由的定制自己的网络结构。


```python
class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
```

#### （2）Algorithm
* `Algorithm` 定义了具体的算法来更新前向网络(`Model`)，也就是通过定义损失函数来更新`Model`，和算法相关的计算都放在`algorithm`中。


```python
# from parl.algorithms import DQN # 也可以直接从parl库中导入DQN算法

class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

```

#### （3）Agent
* `Agent` 负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也一般定义在这里。


```python
class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost
```

### Step5 ReplayMemory
* 经验池：用于存储多条经验，实现 经验回放。


```python
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

```

### Step6 Training && Test（训练&&测试）


```python
# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


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

```

### Step7 创建环境和Agent，创建经验池，启动训练，保存模型


```python
env = gym.make('CartPole-v0')  # CartPole-v0: 预期最后一次评估总分 > 180（最大值是200）
action_dim = env.action_space.n  # CartPole-v0: 2
obs_shape = env.observation_space.shape  # CartPole-v0: (4,)

rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 2000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm)
        episode += 1

    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))

# 训练结束，保存模型
save_path = './dqn_model.ckpt'
agent.save(save_path)
```

    [32m[06-09 16:10:33 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-09 16:10:33 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-09 16:10:33 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-09 16:10:35 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:50    e_greed:0.09926899999999927   test_reward:9.6
    [32m[06-09 16:10:37 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:100    e_greed:0.09872899999999873   test_reward:12.4
    [32m[06-09 16:10:40 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:150    e_greed:0.0981919999999982   test_reward:11.4
    [32m[06-09 16:10:42 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:200    e_greed:0.09762199999999763   test_reward:9.6
    [32m[06-09 16:10:44 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:250    e_greed:0.09710399999999711   test_reward:9.2
    [32m[06-09 16:10:45 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:300    e_greed:0.09655699999999656   test_reward:10.2
    [32m[06-09 16:10:47 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:350    e_greed:0.09595199999999596   test_reward:10.0
    [32m[06-09 16:10:50 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:400    e_greed:0.09522299999999523   test_reward:9.4
    [32m[06-09 16:11:05 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:450    e_greed:0.09127799999999128   test_reward:119.6
    [32m[06-09 16:11:40 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:500    e_greed:0.08295799999998296   test_reward:200.0
    [32m[06-09 16:12:23 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:550    e_greed:0.07321999999997322   test_reward:161.2
    [32m[06-09 16:13:06 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:600    e_greed:0.06383999999996384   test_reward:172.6
    [32m[06-09 16:13:46 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:650    e_greed:0.054751999999954754   test_reward:139.2
    [32m[06-09 16:14:21 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:700    e_greed:0.04665099999994665   test_reward:127.6
    [32m[06-09 16:14:55 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:750    e_greed:0.03920699999993921   test_reward:134.2
    [32m[06-09 16:15:28 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:800    e_greed:0.03157199999993157   test_reward:181.2
    [32m[06-09 16:16:06 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:850    e_greed:0.023422999999923422   test_reward:114.4
    [32m[06-09 16:16:46 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:900    e_greed:0.015071999999916031   test_reward:193.8
    [32m[06-09 16:17:26 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:950    e_greed:0.01   test_reward:158.6
    [32m[06-09 16:17:55 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1000    e_greed:0.01   test_reward:175.4
    [32m[06-09 16:18:28 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1050    e_greed:0.01   test_reward:136.8
    [32m[06-09 16:18:55 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1100    e_greed:0.01   test_reward:140.6
    [32m[06-09 16:19:30 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1150    e_greed:0.01   test_reward:76.4
    [32m[06-09 16:19:59 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1200    e_greed:0.01   test_reward:195.0
    [32m[06-09 16:20:39 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1250    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:21:25 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1300    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:22:06 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1350    e_greed:0.01   test_reward:131.2
    [32m[06-09 16:22:51 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1400    e_greed:0.01   test_reward:193.2
    [32m[06-09 16:23:36 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1450    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:24:17 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1500    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:24:58 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1550    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:25:42 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1600    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:26:09 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1650    e_greed:0.01   test_reward:70.4
    [32m[06-09 16:26:20 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1700    e_greed:0.01   test_reward:118.0
    [32m[06-09 16:27:03 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1750    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:27:49 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1800    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:28:35 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1850    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:29:08 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1900    e_greed:0.01   test_reward:114.2
    [32m[06-09 16:29:40 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:1950    e_greed:0.01   test_reward:200.0
    [32m[06-09 16:30:24 MainThread @<ipython-input-14-1f01c261665f>:39][0m episode:2000    e_greed:0.01   test_reward:200.0

