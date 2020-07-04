# Lesson 5 连续动作空间上求解RL——DDPG
## 1. DDPG简介
* `DDPG`的提出动机其实是为了让`DQN`可以扩展到连续的动作空间。
* `DDPG`借鉴了`DQN`的两个技巧：经验回放 和 固定`Q`网络。
* `DDPG`使用策略网络直接输出确定性动作。
* `DDPG`使用了`Actor-Critic`的架构。

## 2. DDPG实践
* 使用`DDPG`解决连续控制版本的`CartPole`问题，给小车一个力（连续量）使得车上的摆杆倒立起来。

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
import gym
import numpy as np
from copy import deepcopy

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
```

### Step3 设置超参数


```python
ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate

GAMMA = 0.99      # reward 的衰减因子
TAU = 0.001       # 软更新的系数
MEMORY_SIZE = int(1e6)                  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 128
REWARD_SCALE = 0.1   # reward 缩放系数
NOISE = 0.05         # 动作噪声方差

TRAIN_EPISODE = 6000 # 训练的总episode数
```

### Step4 搭建Model、Algorithm、Agent架构
* `Agent`把产生的数据传给`algorithm`，`algorithm`根据`model`的模型结构计算出`Loss`，使用`SGD`或者其他优化器不断的优化，`PARL`这种架构可以很方便的应用在各类深度强化学习问题中。
#### （1）Model
`Model`用来定义前向(`Forward`)网络，用户可以自由的定制自己的网络结构


```python
class Model(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hid = self.fc1(obs)
        means = self.fc2(hid)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q

```

#### （2）Algorithm
* `Algorithm` 定义了具体的算法来更新前向网络(`Model`)，也就是通过定义损失函数来更新`Model`，和算法相关的计算都放在`algorithm`中。


```python
# from parl.algorithms import DDPG # 也可以直接从parl库中快速引入DDPG算法，无需自己重新写算法

class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """  DDPG algorithm
        
        Args:
            model (parl.Model): actor and critic 的前向网络.
                                model 必须实现 get_actor_params() 方法.
            gamma (float): reward的衰减因子.
            tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
            actor_lr (float): actor 的学习率
            critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 用DDPG算法更新 actor 和 critic
        """
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        return actor_cost, critic_cost

    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        cost = layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        Q = self.model.value(obs, action)
        cost = layers.square_error_cost(Q, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，可设置软更新参数
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)

```

#### （3）Agent
* `Agent`负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也一般定义在这里。


```python
class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        act = np.squeeze(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost

```

### env.py
#### 连续控制版本的CartPole环境 
* 该环境代码与算法无关，可忽略不看


```python
# env.py
# Continuous version of Cartpole

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        action = np.expand_dims(action, 0)
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()

```

## replay_memory.py
#### 经验池 ReplayMemory
* 与`DQN`的`replay_mamory.py`代码一致


```python
# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

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

### Step5 Training && Test（训练&&测试）


```python
def run_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))

        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)

        next_obs, reward, done, info = env.step(action)

        action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward

        if done or steps >= 200:
            break
    return total_reward


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action, -1.0, 1.0)

            steps += 1
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()
            if done or steps >= 200:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

```

### Step6 创建环境和Agent，创建经验池，启动训练，保存模型


```python
# 创建环境
env = ContinuousCartPoleEnv()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 使用PARL框架创建agent
model = Model(act_dim)
algorithm = DDPG(
    model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = Agent(algorithm, obs_dim, act_dim)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)
# 往经验池中预存数据
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(agent, env, rpm)

episode = 0
while episode < TRAIN_EPISODE:
    for i in range(50):
        total_reward = run_episode(agent, env, rpm)
        episode += 1

    eval_reward = evaluate(env, agent, render=False)
    logger.info('episode:{}    test_reward:{}'.format(
        episode, eval_reward))
```

    [32m[06-11 16:26:59 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-11 16:26:59 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-11 16:26:59 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-11 16:28:44 MainThread @machine_info.py:84][0m Cannot find available GPU devices, using CPU now.
    [32m[06-11 16:28:46 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:50    test_reward:6.6
    [32m[06-11 16:28:48 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:100    test_reward:5.8
    [32m[06-11 16:28:49 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:150    test_reward:6.0
    [32m[06-11 16:28:50 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:200    test_reward:6.0
    [32m[06-11 16:28:52 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:250    test_reward:5.8
    [32m[06-11 16:28:53 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:300    test_reward:5.8
    [32m[06-11 16:28:55 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:350    test_reward:5.8
    [32m[06-11 16:28:56 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:400    test_reward:6.2
    [32m[06-11 16:28:58 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:450    test_reward:5.8
    [32m[06-11 16:28:59 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:500    test_reward:6.0
    [32m[06-11 16:29:01 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:550    test_reward:5.8
    [32m[06-11 16:29:02 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:600    test_reward:5.8
    [32m[06-11 16:29:04 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:650    test_reward:5.8
    [32m[06-11 16:29:05 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:700    test_reward:5.8
    [32m[06-11 16:29:06 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:750    test_reward:5.8
    [32m[06-11 16:29:08 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:800    test_reward:5.8
    [32m[06-11 16:29:09 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:850    test_reward:5.8
    [32m[06-11 16:29:11 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:900    test_reward:6.0
    [32m[06-11 16:29:12 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:950    test_reward:5.8
    [32m[06-11 16:29:14 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1000    test_reward:6.0
    [32m[06-11 16:29:15 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1050    test_reward:6.0
    [32m[06-11 16:29:17 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1100    test_reward:5.6
    [32m[06-11 16:29:18 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1150    test_reward:6.2
    [32m[06-11 16:29:19 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1200    test_reward:5.8
    [32m[06-11 16:29:21 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1250    test_reward:6.0
    [32m[06-11 16:29:22 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1300    test_reward:6.0
    [32m[06-11 16:29:24 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1350    test_reward:5.6
    [32m[06-11 16:29:25 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1400    test_reward:5.8
    [32m[06-11 16:29:27 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1450    test_reward:6.0
    [32m[06-11 16:29:28 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1500    test_reward:5.8
    [32m[06-11 16:29:30 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1550    test_reward:6.0
    [32m[06-11 16:29:31 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1600    test_reward:5.8
    [32m[06-11 16:29:32 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1650    test_reward:6.0
    [32m[06-11 16:29:34 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1700    test_reward:5.6
    [32m[06-11 16:29:35 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1750    test_reward:6.0
    [32m[06-11 16:29:37 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1800    test_reward:6.0
    [32m[06-11 16:29:39 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1850    test_reward:5.8
    [32m[06-11 16:29:40 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1900    test_reward:6.0
    [32m[06-11 16:29:42 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:1950    test_reward:5.6
    [32m[06-11 16:29:43 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2000    test_reward:5.8
    [32m[06-11 16:29:45 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2050    test_reward:6.0
    [32m[06-11 16:29:46 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2100    test_reward:5.8
    [32m[06-11 16:29:48 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2150    test_reward:5.6
    [32m[06-11 16:29:49 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2200    test_reward:5.6
    [32m[06-11 16:29:50 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2250    test_reward:6.0
    [32m[06-11 16:29:52 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2300    test_reward:5.8
    [32m[06-11 16:29:53 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2350    test_reward:6.0
    [32m[06-11 16:29:55 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2400    test_reward:5.8
    [32m[06-11 16:29:56 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2450    test_reward:6.0
    [32m[06-11 16:29:58 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2500    test_reward:6.0
    [32m[06-11 16:29:59 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2550    test_reward:5.8
    [32m[06-11 16:30:01 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2600    test_reward:5.8
    [32m[06-11 16:30:02 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2650    test_reward:5.8
    [32m[06-11 16:30:04 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2700    test_reward:5.4
    [32m[06-11 16:30:05 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2750    test_reward:6.0
    [32m[06-11 16:30:07 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2800    test_reward:6.0
    [32m[06-11 16:30:08 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2850    test_reward:6.0
    [32m[06-11 16:30:10 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2900    test_reward:5.8
    [32m[06-11 16:30:11 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:2950    test_reward:6.2
    [32m[06-11 16:30:13 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3000    test_reward:5.8
    [32m[06-11 16:30:14 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3050    test_reward:5.8
    [32m[06-11 16:30:16 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3100    test_reward:6.0
    [32m[06-11 16:30:17 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3150    test_reward:5.8
    [32m[06-11 16:30:19 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3200    test_reward:5.8
    [32m[06-11 16:30:20 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3250    test_reward:5.8
    [32m[06-11 16:30:22 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3300    test_reward:6.0
    [32m[06-11 16:30:23 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3350    test_reward:5.8
    [32m[06-11 16:30:25 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3400    test_reward:5.6
    [32m[06-11 16:30:26 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3450    test_reward:6.0
    [32m[06-11 16:30:28 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3500    test_reward:5.8
    [32m[06-11 16:30:29 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3550    test_reward:6.0
    [32m[06-11 16:30:31 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3600    test_reward:5.8
    [32m[06-11 16:30:32 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3650    test_reward:5.8
    [32m[06-11 16:30:34 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3700    test_reward:6.0
    [32m[06-11 16:30:36 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3750    test_reward:6.0
    [32m[06-11 16:30:37 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3800    test_reward:6.0
    [32m[06-11 16:30:39 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3850    test_reward:5.6
    [32m[06-11 16:30:40 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3900    test_reward:6.0
    [32m[06-11 16:30:42 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:3950    test_reward:5.6
    [32m[06-11 16:30:43 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4000    test_reward:5.6
    [32m[06-11 16:30:50 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4050    test_reward:56.0
    [32m[06-11 16:31:19 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4100    test_reward:200.0
    [32m[06-11 16:32:11 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4150    test_reward:200.0
    [32m[06-11 16:33:08 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4200    test_reward:200.0
    [32m[06-11 16:34:06 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4250    test_reward:200.0
    [32m[06-11 16:35:05 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4300    test_reward:200.0
    [32m[06-11 16:36:04 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4350    test_reward:200.0
    [32m[06-11 16:37:04 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4400    test_reward:200.0
    [32m[06-11 16:38:03 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4450    test_reward:200.0
    [32m[06-11 16:39:02 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4500    test_reward:200.0
    [32m[06-11 16:40:03 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4550    test_reward:200.0
    [32m[06-11 16:40:58 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4600    test_reward:141.2
    [32m[06-11 16:41:50 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4650    test_reward:200.0
    [32m[06-11 16:42:53 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4700    test_reward:200.0
    [32m[06-11 16:43:54 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4750    test_reward:200.0
    [32m[06-11 16:44:55 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4800    test_reward:200.0
    [32m[06-11 16:45:57 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4850    test_reward:200.0
    [32m[06-11 16:46:57 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4900    test_reward:200.0
    [32m[06-11 16:47:58 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:4950    test_reward:200.0
    [32m[06-11 16:48:57 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5000    test_reward:200.0
    [32m[06-11 16:49:58 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5050    test_reward:200.0
    [32m[06-11 16:50:56 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5100    test_reward:200.0
    [32m[06-11 16:52:03 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5150    test_reward:200.0
    [32m[06-11 16:53:07 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5200    test_reward:200.0
    [32m[06-11 16:54:15 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5250    test_reward:200.0
    [32m[06-11 16:55:18 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5300    test_reward:200.0
    [32m[06-11 16:56:28 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5350    test_reward:200.0
    [32m[06-11 16:57:34 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5400    test_reward:198.4
    [32m[06-11 16:58:43 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5450    test_reward:188.4
    [32m[06-11 16:59:51 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5500    test_reward:198.8
    [32m[06-11 17:00:58 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5550    test_reward:200.0
    [32m[06-11 17:02:10 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5600    test_reward:200.0
    [32m[06-11 17:03:18 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5650    test_reward:200.0
    [32m[06-11 17:04:31 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5700    test_reward:200.0
    [32m[06-11 17:05:44 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5750    test_reward:200.0
    [32m[06-11 17:06:53 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5800    test_reward:200.0
    [32m[06-11 17:08:09 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5850    test_reward:200.0
    [32m[06-11 17:09:23 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5900    test_reward:200.0
    [32m[06-11 17:10:32 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:5950    test_reward:200.0
    [32m[06-11 17:11:52 MainThread @<ipython-input-10-f4e0a11a1638>:27][0m episode:6000    test_reward:200.0

