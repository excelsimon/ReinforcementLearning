# Lesson2 表格型方法—— Q-learning
## 1. Q-learning简介
* `Q-learning`也是采用`Q`表格的方式存储`Q`值（状态动作价值），决策部分与`Sarsa`是一样的，采用`ε-greedy`方式增加探索。
* `Q-learning`跟`Sarsa`不一样的地方是更新`Q`表格的方式。
    * `Sarsa`是`on-policy`的更新方式，先做出动作再更新。
    * `Q-learning`是`off-policy`的更新方式，更新`learn()`时无需获取下一步实际做出的动作`next_action`，并假设下一步动作是取最大`Q`值的动作。
* `Q-learning`的更新公式为：

![](https://ai-studio-static-online.cdn.bcebos.com/38158582039041edad0a5a704ba792d0e464f2eb8a394577bf88051cc52d6b66)

## 2. Q-learning实战
* 使用`Q-learning`解决悬崖问题，找到绕过悬崖通往终端的最短路径。

### Step1 安装依赖



```python
!pip install gym
```

### Step2 导入依赖


```python
import gym
import time
import numpy as np 
```

### Step3 Agent
* `Agent`是和环境`environment`交互的主体。
* `predict()`方法：输入观察值`observation`（或者说状态`state`），输出动作值
* `sample()`方法：在`predict()`方法基础上使用`ε-greedy`增加探索
* `learn()`方法：输入训练数据，完成一轮`Q`表格的更新


```python
class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon): #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n) #有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :]) # Q-learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q) # 修正q

    # 把 Q表格 的数据保存到文件中
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')
    
    # 从文件中读取数据到 Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')

```

### Step4 Training && Test（训练&&测试）
* `run_episode()`：`agent`在一个`episode`中训练的过程，使用`agent.sample()`与环境交互，使用`agent.learn()`训练`Q`表格。
* `test_episode()`：`agent`在一个`episode`中测试效果，评估目前的`agent`能在一个`episode`中拿到多少总`reward`。


```python
# train.py

def run_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs) # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action) # 与环境进行一个交互
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.5)
        # env.render()
        if done:
            break
    return total_reward

```

### Step5 创建环境和Agent，启动训练


```python
# 使用gym创建悬崖环境
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

# 创建一个agent实例，输入超参数
agent = QLearningAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.9,
    e_greed=0.1)


# 训练500个episode，打印每个episode的分数
for episode in range(500):
    ep_reward, ep_steps = run_episode(env, agent, False)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 全部训练结束，查看算法效果
test_reward = test_episode(env, agent)
print('test reward = %.1f' % (test_reward))
```

    Episode 0: steps = 262 , reward = -955.0
    Episode 1: steps = 686 , reward = -1577.0
    Episode 2: steps = 381 , reward = -777.0
    Episode 3: steps = 352 , reward = -748.0
    Episode 4: steps = 201 , reward = -201.0
    Episode 5: steps = 279 , reward = -576.0
    Episode 6: steps = 241 , reward = -439.0
    Episode 7: steps = 72 , reward = -72.0
    Episode 8: steps = 149 , reward = -347.0
    Episode 9: steps = 176 , reward = -275.0
    Episode 10: steps = 119 , reward = -119.0
    Episode 11: steps = 276 , reward = -672.0
    Episode 12: steps = 186 , reward = -285.0
    Episode 13: steps = 90 , reward = -288.0
    Episode 14: steps = 58 , reward = -58.0
    Episode 15: steps = 135 , reward = -135.0
    Episode 16: steps = 83 , reward = -182.0
    Episode 17: steps = 184 , reward = -580.0
    Episode 18: steps = 50 , reward = -50.0
    Episode 19: steps = 114 , reward = -114.0
    Episode 20: steps = 57 , reward = -156.0
    Episode 21: steps = 37 , reward = -37.0
    Episode 22: steps = 55 , reward = -55.0
    Episode 23: steps = 162 , reward = -261.0
    Episode 24: steps = 31 , reward = -31.0
    Episode 25: steps = 57 , reward = -57.0
    Episode 26: steps = 126 , reward = -225.0
    Episode 27: steps = 71 , reward = -269.0
    Episode 28: steps = 106 , reward = -106.0
    Episode 29: steps = 120 , reward = -318.0
    Episode 30: steps = 133 , reward = -232.0
    Episode 31: steps = 39 , reward = -39.0
    Episode 32: steps = 49 , reward = -49.0
    Episode 33: steps = 55 , reward = -55.0
    Episode 34: steps = 54 , reward = -54.0
    Episode 35: steps = 83 , reward = -182.0
    Episode 36: steps = 76 , reward = -175.0
    Episode 37: steps = 100 , reward = -199.0
    Episode 38: steps = 32 , reward = -32.0
    Episode 39: steps = 45 , reward = -45.0
    Episode 40: steps = 73 , reward = -172.0
    Episode 41: steps = 100 , reward = -199.0
    Episode 42: steps = 25 , reward = -25.0
    Episode 43: steps = 88 , reward = -88.0
    Episode 44: steps = 54 , reward = -54.0
    Episode 45: steps = 50 , reward = -50.0
    Episode 46: steps = 72 , reward = -72.0
    Episode 47: steps = 63 , reward = -261.0
    Episode 48: steps = 52 , reward = -52.0
    Episode 49: steps = 61 , reward = -61.0
    Episode 50: steps = 55 , reward = -55.0
    Episode 51: steps = 128 , reward = -326.0
    Episode 52: steps = 78 , reward = -177.0
    Episode 53: steps = 32 , reward = -131.0
    Episode 54: steps = 51 , reward = -51.0
    Episode 55: steps = 37 , reward = -37.0
    Episode 56: steps = 61 , reward = -61.0
    Episode 57: steps = 77 , reward = -275.0
    Episode 58: steps = 66 , reward = -66.0
    Episode 59: steps = 37 , reward = -37.0
    Episode 60: steps = 43 , reward = -43.0
    Episode 61: steps = 91 , reward = -190.0
    Episode 62: steps = 30 , reward = -30.0
    Episode 63: steps = 36 , reward = -36.0
    Episode 64: steps = 45 , reward = -45.0
    Episode 65: steps = 51 , reward = -51.0
    Episode 66: steps = 45 , reward = -144.0
    Episode 67: steps = 57 , reward = -57.0
    Episode 68: steps = 66 , reward = -66.0
    Episode 69: steps = 51 , reward = -51.0
    Episode 70: steps = 50 , reward = -50.0
    Episode 71: steps = 19 , reward = -19.0
    Episode 72: steps = 55 , reward = -55.0
    Episode 73: steps = 39 , reward = -39.0
    Episode 74: steps = 35 , reward = -35.0
    Episode 75: steps = 28 , reward = -28.0
    Episode 76: steps = 50 , reward = -50.0
    Episode 77: steps = 31 , reward = -130.0
    Episode 78: steps = 62 , reward = -62.0
    Episode 79: steps = 34 , reward = -34.0
    Episode 80: steps = 22 , reward = -22.0
    Episode 81: steps = 82 , reward = -82.0
    Episode 82: steps = 22 , reward = -22.0
    Episode 83: steps = 35 , reward = -35.0
    Episode 84: steps = 39 , reward = -39.0
    Episode 85: steps = 35 , reward = -35.0
    Episode 86: steps = 41 , reward = -41.0
    Episode 87: steps = 35 , reward = -35.0
    Episode 88: steps = 16 , reward = -16.0
    Episode 89: steps = 33 , reward = -33.0
    Episode 90: steps = 73 , reward = -73.0
    Episode 91: steps = 35 , reward = -35.0
    Episode 92: steps = 38 , reward = -38.0
    Episode 93: steps = 84 , reward = -282.0
    Episode 94: steps = 40 , reward = -40.0
    Episode 95: steps = 30 , reward = -30.0
    Episode 96: steps = 90 , reward = -189.0
    Episode 97: steps = 57 , reward = -156.0
    Episode 98: steps = 28 , reward = -28.0
    Episode 99: steps = 46 , reward = -46.0
    Episode 100: steps = 24 , reward = -24.0
    Episode 101: steps = 44 , reward = -44.0
    Episode 102: steps = 30 , reward = -30.0
    Episode 103: steps = 23 , reward = -23.0
    Episode 104: steps = 49 , reward = -148.0
    Episode 105: steps = 22 , reward = -22.0
    Episode 106: steps = 50 , reward = -149.0
    Episode 107: steps = 63 , reward = -261.0
    Episode 108: steps = 36 , reward = -36.0
    Episode 109: steps = 49 , reward = -49.0
    Episode 110: steps = 40 , reward = -139.0
    Episode 111: steps = 60 , reward = -159.0
    Episode 112: steps = 19 , reward = -19.0
    Episode 113: steps = 54 , reward = -54.0
    Episode 114: steps = 26 , reward = -26.0
    Episode 115: steps = 17 , reward = -17.0
    Episode 116: steps = 35 , reward = -35.0
    Episode 117: steps = 53 , reward = -152.0
    Episode 118: steps = 27 , reward = -27.0
    Episode 119: steps = 31 , reward = -31.0
    Episode 120: steps = 20 , reward = -20.0
    Episode 121: steps = 61 , reward = -160.0
    Episode 122: steps = 43 , reward = -43.0
    Episode 123: steps = 28 , reward = -28.0
    Episode 124: steps = 54 , reward = -153.0
    Episode 125: steps = 26 , reward = -26.0
    Episode 126: steps = 23 , reward = -23.0
    Episode 127: steps = 15 , reward = -15.0
    Episode 128: steps = 37 , reward = -37.0
    Episode 129: steps = 28 , reward = -28.0
    Episode 130: steps = 37 , reward = -37.0
    Episode 131: steps = 43 , reward = -142.0
    Episode 132: steps = 20 , reward = -20.0
    Episode 133: steps = 44 , reward = -44.0
    Episode 134: steps = 19 , reward = -19.0
    Episode 135: steps = 36 , reward = -36.0
    Episode 136: steps = 31 , reward = -31.0
    Episode 137: steps = 45 , reward = -45.0
    Episode 138: steps = 50 , reward = -149.0
    Episode 139: steps = 23 , reward = -23.0
    Episode 140: steps = 24 , reward = -24.0
    Episode 141: steps = 39 , reward = -39.0
    Episode 142: steps = 18 , reward = -18.0
    Episode 143: steps = 20 , reward = -20.0
    Episode 144: steps = 31 , reward = -31.0
    Episode 145: steps = 23 , reward = -23.0
    Episode 146: steps = 40 , reward = -139.0
    Episode 147: steps = 24 , reward = -24.0
    Episode 148: steps = 36 , reward = -36.0
    Episode 149: steps = 24 , reward = -24.0
    Episode 150: steps = 23 , reward = -23.0
    Episode 151: steps = 20 , reward = -20.0
    Episode 152: steps = 21 , reward = -21.0
    Episode 153: steps = 43 , reward = -43.0
    Episode 154: steps = 28 , reward = -28.0
    Episode 155: steps = 20 , reward = -119.0
    Episode 156: steps = 41 , reward = -140.0
    Episode 157: steps = 18 , reward = -18.0
    Episode 158: steps = 25 , reward = -25.0
    Episode 159: steps = 24 , reward = -24.0
    Episode 160: steps = 31 , reward = -31.0
    Episode 161: steps = 28 , reward = -127.0
    Episode 162: steps = 25 , reward = -25.0
    Episode 163: steps = 19 , reward = -19.0
    Episode 164: steps = 47 , reward = -146.0
    Episode 165: steps = 21 , reward = -21.0
    Episode 166: steps = 21 , reward = -21.0
    Episode 167: steps = 18 , reward = -18.0
    Episode 168: steps = 23 , reward = -23.0
    Episode 169: steps = 31 , reward = -31.0
    Episode 170: steps = 19 , reward = -19.0
    Episode 171: steps = 20 , reward = -20.0
    Episode 172: steps = 48 , reward = -147.0
    Episode 173: steps = 19 , reward = -19.0
    Episode 174: steps = 39 , reward = -39.0
    Episode 175: steps = 29 , reward = -128.0
    Episode 176: steps = 47 , reward = -47.0
    Episode 177: steps = 19 , reward = -19.0
    Episode 178: steps = 25 , reward = -25.0
    Episode 179: steps = 22 , reward = -22.0
    Episode 180: steps = 27 , reward = -126.0
    Episode 181: steps = 47 , reward = -146.0
    Episode 182: steps = 17 , reward = -17.0
    Episode 183: steps = 60 , reward = -258.0
    Episode 184: steps = 28 , reward = -127.0
    Episode 185: steps = 13 , reward = -13.0
    Episode 186: steps = 15 , reward = -15.0
    Episode 187: steps = 33 , reward = -33.0
    Episode 188: steps = 21 , reward = -21.0
    Episode 189: steps = 36 , reward = -234.0
    Episode 190: steps = 30 , reward = -30.0
    Episode 191: steps = 18 , reward = -18.0
    Episode 192: steps = 32 , reward = -32.0
    Episode 193: steps = 20 , reward = -218.0
    Episode 194: steps = 30 , reward = -30.0
    Episode 195: steps = 20 , reward = -119.0
    Episode 196: steps = 18 , reward = -18.0
    Episode 197: steps = 16 , reward = -16.0
    Episode 198: steps = 27 , reward = -27.0
    Episode 199: steps = 18 , reward = -18.0
    Episode 200: steps = 23 , reward = -23.0
    Episode 201: steps = 42 , reward = -240.0
    Episode 202: steps = 34 , reward = -133.0
    Episode 203: steps = 24 , reward = -24.0
    Episode 204: steps = 17 , reward = -17.0
    Episode 205: steps = 21 , reward = -21.0
    Episode 206: steps = 25 , reward = -25.0
    Episode 207: steps = 42 , reward = -339.0
    Episode 208: steps = 21 , reward = -21.0
    Episode 209: steps = 13 , reward = -13.0
    Episode 210: steps = 25 , reward = -25.0
    Episode 211: steps = 17 , reward = -17.0
    Episode 212: steps = 36 , reward = -36.0
    Episode 213: steps = 26 , reward = -26.0
    Episode 214: steps = 15 , reward = -15.0
    Episode 215: steps = 36 , reward = -135.0
    Episode 216: steps = 13 , reward = -13.0
    Episode 217: steps = 16 , reward = -16.0
    Episode 218: steps = 20 , reward = -20.0
    Episode 219: steps = 20 , reward = -20.0
    Episode 220: steps = 31 , reward = -130.0
    Episode 221: steps = 15 , reward = -15.0
    Episode 222: steps = 40 , reward = -139.0
    Episode 223: steps = 18 , reward = -18.0
    Episode 224: steps = 17 , reward = -17.0
    Episode 225: steps = 29 , reward = -29.0
    Episode 226: steps = 21 , reward = -21.0
    Episode 227: steps = 20 , reward = -20.0
    Episode 228: steps = 13 , reward = -13.0
    Episode 229: steps = 20 , reward = -20.0
    Episode 230: steps = 16 , reward = -16.0
    Episode 231: steps = 17 , reward = -17.0
    Episode 232: steps = 32 , reward = -131.0
    Episode 233: steps = 13 , reward = -13.0
    Episode 234: steps = 16 , reward = -16.0
    Episode 235: steps = 25 , reward = -124.0
    Episode 236: steps = 13 , reward = -13.0
    Episode 237: steps = 27 , reward = -126.0
    Episode 238: steps = 29 , reward = -128.0
    Episode 239: steps = 13 , reward = -13.0
    Episode 240: steps = 43 , reward = -241.0
    Episode 241: steps = 19 , reward = -19.0
    Episode 242: steps = 14 , reward = -14.0
    Episode 243: steps = 27 , reward = -126.0
    Episode 244: steps = 33 , reward = -132.0
    Episode 245: steps = 18 , reward = -18.0
    Episode 246: steps = 28 , reward = -127.0
    Episode 247: steps = 23 , reward = -23.0
    Episode 248: steps = 13 , reward = -13.0
    Episode 249: steps = 15 , reward = -15.0
    Episode 250: steps = 17 , reward = -17.0
    Episode 251: steps = 30 , reward = -30.0
    Episode 252: steps = 16 , reward = -16.0
    Episode 253: steps = 20 , reward = -20.0
    Episode 254: steps = 19 , reward = -19.0
    Episode 255: steps = 13 , reward = -13.0
    Episode 256: steps = 13 , reward = -13.0
    Episode 257: steps = 27 , reward = -126.0
    Episode 258: steps = 18 , reward = -18.0
    Episode 259: steps = 22 , reward = -22.0
    Episode 260: steps = 15 , reward = -15.0
    Episode 261: steps = 23 , reward = -23.0
    Episode 262: steps = 13 , reward = -13.0
    Episode 263: steps = 17 , reward = -116.0
    Episode 264: steps = 23 , reward = -122.0
    Episode 265: steps = 13 , reward = -13.0
    Episode 266: steps = 19 , reward = -118.0
    Episode 267: steps = 16 , reward = -115.0
    Episode 268: steps = 13 , reward = -13.0
    Episode 269: steps = 39 , reward = -237.0
    Episode 270: steps = 20 , reward = -119.0
    Episode 271: steps = 28 , reward = -127.0
    Episode 272: steps = 24 , reward = -123.0
    Episode 273: steps = 13 , reward = -13.0
    Episode 274: steps = 43 , reward = -142.0
    Episode 275: steps = 14 , reward = -14.0
    Episode 276: steps = 15 , reward = -15.0
    Episode 277: steps = 22 , reward = -121.0
    Episode 278: steps = 23 , reward = -122.0
    Episode 279: steps = 13 , reward = -13.0
    Episode 280: steps = 14 , reward = -14.0
    Episode 281: steps = 28 , reward = -226.0
    Episode 282: steps = 15 , reward = -15.0
    Episode 283: steps = 20 , reward = -20.0
    Episode 284: steps = 13 , reward = -13.0
    Episode 285: steps = 17 , reward = -17.0
    Episode 286: steps = 30 , reward = -30.0
    Episode 287: steps = 17 , reward = -17.0
    Episode 288: steps = 29 , reward = -128.0
    Episode 289: steps = 15 , reward = -15.0
    Episode 290: steps = 17 , reward = -17.0
    Episode 291: steps = 13 , reward = -13.0
    Episode 292: steps = 19 , reward = -19.0
    Episode 293: steps = 24 , reward = -222.0
    Episode 294: steps = 20 , reward = -119.0
    Episode 295: steps = 13 , reward = -13.0
    Episode 296: steps = 19 , reward = -118.0
    Episode 297: steps = 13 , reward = -13.0
    Episode 298: steps = 15 , reward = -15.0
    Episode 299: steps = 17 , reward = -17.0
    Episode 300: steps = 24 , reward = -123.0
    Episode 301: steps = 19 , reward = -19.0
    Episode 302: steps = 26 , reward = -26.0
    Episode 303: steps = 14 , reward = -14.0
    Episode 304: steps = 13 , reward = -13.0
    Episode 305: steps = 15 , reward = -15.0
    Episode 306: steps = 13 , reward = -13.0
    Episode 307: steps = 15 , reward = -15.0
    Episode 308: steps = 13 , reward = -13.0
    Episode 309: steps = 19 , reward = -19.0
    Episode 310: steps = 15 , reward = -15.0
    Episode 311: steps = 13 , reward = -13.0
    Episode 312: steps = 30 , reward = -30.0
    Episode 313: steps = 15 , reward = -15.0
    Episode 314: steps = 39 , reward = -336.0
    Episode 315: steps = 20 , reward = -119.0
    Episode 316: steps = 14 , reward = -14.0
    Episode 317: steps = 13 , reward = -13.0
    Episode 318: steps = 13 , reward = -13.0
    Episode 319: steps = 27 , reward = -126.0
    Episode 320: steps = 24 , reward = -123.0
    Episode 321: steps = 13 , reward = -13.0
    Episode 322: steps = 15 , reward = -15.0
    Episode 323: steps = 25 , reward = -124.0
    Episode 324: steps = 13 , reward = -13.0
    Episode 325: steps = 20 , reward = -119.0
    Episode 326: steps = 13 , reward = -13.0
    Episode 327: steps = 14 , reward = -14.0
    Episode 328: steps = 17 , reward = -17.0
    Episode 329: steps = 24 , reward = -24.0
    Episode 330: steps = 17 , reward = -17.0
    Episode 331: steps = 17 , reward = -17.0
    Episode 332: steps = 17 , reward = -17.0
    Episode 333: steps = 15 , reward = -15.0
    Episode 334: steps = 13 , reward = -13.0
    Episode 335: steps = 14 , reward = -14.0
    Episode 336: steps = 16 , reward = -16.0
    Episode 337: steps = 31 , reward = -130.0
    Episode 338: steps = 15 , reward = -15.0
    Episode 339: steps = 15 , reward = -15.0
    Episode 340: steps = 13 , reward = -13.0
    Episode 341: steps = 25 , reward = -124.0
    Episode 342: steps = 13 , reward = -13.0
    Episode 343: steps = 13 , reward = -13.0
    Episode 344: steps = 13 , reward = -13.0
    Episode 345: steps = 13 , reward = -13.0
    Episode 346: steps = 13 , reward = -13.0
    Episode 347: steps = 13 , reward = -13.0
    Episode 348: steps = 34 , reward = -133.0
    Episode 349: steps = 15 , reward = -15.0
    Episode 350: steps = 15 , reward = -15.0
    Episode 351: steps = 13 , reward = -13.0
    Episode 352: steps = 19 , reward = -118.0
    Episode 353: steps = 13 , reward = -13.0
    Episode 354: steps = 13 , reward = -13.0
    Episode 355: steps = 13 , reward = -13.0
    Episode 356: steps = 20 , reward = -119.0
    Episode 357: steps = 13 , reward = -13.0
    Episode 358: steps = 24 , reward = -123.0
    Episode 359: steps = 13 , reward = -13.0
    Episode 360: steps = 31 , reward = -328.0
    Episode 361: steps = 15 , reward = -15.0
    Episode 362: steps = 15 , reward = -15.0
    Episode 363: steps = 15 , reward = -15.0
    Episode 364: steps = 15 , reward = -15.0
    Episode 365: steps = 13 , reward = -13.0
    Episode 366: steps = 13 , reward = -13.0
    Episode 367: steps = 13 , reward = -13.0
    Episode 368: steps = 13 , reward = -13.0
    Episode 369: steps = 13 , reward = -13.0
    Episode 370: steps = 17 , reward = -17.0
    Episode 371: steps = 14 , reward = -14.0
    Episode 372: steps = 20 , reward = -218.0
    Episode 373: steps = 15 , reward = -15.0
    Episode 374: steps = 19 , reward = -19.0
    Episode 375: steps = 13 , reward = -13.0
    Episode 376: steps = 13 , reward = -13.0
    Episode 377: steps = 13 , reward = -13.0
    Episode 378: steps = 26 , reward = -125.0
    Episode 379: steps = 13 , reward = -13.0
    Episode 380: steps = 13 , reward = -13.0
    Episode 381: steps = 13 , reward = -13.0
    Episode 382: steps = 19 , reward = -19.0
    Episode 383: steps = 13 , reward = -13.0
    Episode 384: steps = 13 , reward = -13.0
    Episode 385: steps = 13 , reward = -13.0
    Episode 386: steps = 15 , reward = -15.0
    Episode 387: steps = 13 , reward = -13.0
    Episode 388: steps = 13 , reward = -13.0
    Episode 389: steps = 15 , reward = -15.0
    Episode 390: steps = 17 , reward = -17.0
    Episode 391: steps = 13 , reward = -13.0
    Episode 392: steps = 13 , reward = -13.0
    Episode 393: steps = 15 , reward = -15.0
    Episode 394: steps = 41 , reward = -338.0
    Episode 395: steps = 15 , reward = -15.0
    Episode 396: steps = 17 , reward = -17.0
    Episode 397: steps = 33 , reward = -231.0
    Episode 398: steps = 34 , reward = -232.0
    Episode 399: steps = 13 , reward = -13.0
    Episode 400: steps = 19 , reward = -19.0
    Episode 401: steps = 15 , reward = -114.0
    Episode 402: steps = 27 , reward = -225.0
    Episode 403: steps = 13 , reward = -13.0
    Episode 404: steps = 17 , reward = -17.0
    Episode 405: steps = 13 , reward = -13.0
    Episode 406: steps = 13 , reward = -13.0
    Episode 407: steps = 30 , reward = -129.0
    Episode 408: steps = 15 , reward = -15.0
    Episode 409: steps = 15 , reward = -15.0
    Episode 410: steps = 15 , reward = -15.0
    Episode 411: steps = 27 , reward = -27.0
    Episode 412: steps = 17 , reward = -17.0
    Episode 413: steps = 15 , reward = -15.0
    Episode 414: steps = 17 , reward = -17.0
    Episode 415: steps = 13 , reward = -13.0
    Episode 416: steps = 27 , reward = -126.0
    Episode 417: steps = 13 , reward = -13.0
    Episode 418: steps = 15 , reward = -15.0
    Episode 419: steps = 13 , reward = -13.0
    Episode 420: steps = 19 , reward = -19.0
    Episode 421: steps = 13 , reward = -13.0
    Episode 422: steps = 13 , reward = -13.0
    Episode 423: steps = 13 , reward = -13.0
    Episode 424: steps = 13 , reward = -13.0
    Episode 425: steps = 18 , reward = -18.0
    Episode 426: steps = 15 , reward = -15.0
    Episode 427: steps = 13 , reward = -13.0
    Episode 428: steps = 13 , reward = -13.0
    Episode 429: steps = 15 , reward = -15.0
    Episode 430: steps = 26 , reward = -125.0
    Episode 431: steps = 15 , reward = -15.0
    Episode 432: steps = 13 , reward = -13.0
    Episode 433: steps = 15 , reward = -15.0
    Episode 434: steps = 15 , reward = -15.0
    Episode 435: steps = 13 , reward = -13.0
    Episode 436: steps = 17 , reward = -17.0
    Episode 437: steps = 17 , reward = -17.0
    Episode 438: steps = 15 , reward = -15.0
    Episode 439: steps = 15 , reward = -15.0
    Episode 440: steps = 14 , reward = -14.0
    Episode 441: steps = 13 , reward = -13.0
    Episode 442: steps = 13 , reward = -13.0
    Episode 443: steps = 13 , reward = -13.0
    Episode 444: steps = 21 , reward = -21.0
    Episode 445: steps = 13 , reward = -13.0
    Episode 446: steps = 13 , reward = -13.0
    Episode 447: steps = 13 , reward = -13.0
    Episode 448: steps = 13 , reward = -13.0
    Episode 449: steps = 17 , reward = -116.0
    Episode 450: steps = 15 , reward = -15.0
    Episode 451: steps = 13 , reward = -13.0
    Episode 452: steps = 14 , reward = -14.0
    Episode 453: steps = 13 , reward = -13.0
    Episode 454: steps = 13 , reward = -13.0
    Episode 455: steps = 13 , reward = -13.0
    Episode 456: steps = 24 , reward = -123.0
    Episode 457: steps = 17 , reward = -116.0
    Episode 458: steps = 13 , reward = -13.0
    Episode 459: steps = 17 , reward = -17.0
    Episode 460: steps = 18 , reward = -18.0
    Episode 461: steps = 13 , reward = -13.0
    Episode 462: steps = 13 , reward = -13.0
    Episode 463: steps = 21 , reward = -120.0
    Episode 464: steps = 13 , reward = -13.0
    Episode 465: steps = 15 , reward = -15.0
    Episode 466: steps = 17 , reward = -17.0
    Episode 467: steps = 15 , reward = -15.0
    Episode 468: steps = 19 , reward = -118.0
    Episode 469: steps = 14 , reward = -14.0
    Episode 470: steps = 13 , reward = -13.0
    Episode 471: steps = 21 , reward = -21.0
    Episode 472: steps = 15 , reward = -15.0
    Episode 473: steps = 15 , reward = -15.0
    Episode 474: steps = 14 , reward = -14.0
    Episode 475: steps = 13 , reward = -13.0
    Episode 476: steps = 15 , reward = -15.0
    Episode 477: steps = 15 , reward = -15.0
    Episode 478: steps = 15 , reward = -15.0
    Episode 479: steps = 16 , reward = -16.0
    Episode 480: steps = 19 , reward = -118.0
    Episode 481: steps = 15 , reward = -15.0
    Episode 482: steps = 26 , reward = -125.0
    Episode 483: steps = 24 , reward = -123.0
    Episode 484: steps = 14 , reward = -14.0
    Episode 485: steps = 27 , reward = -27.0
    Episode 486: steps = 14 , reward = -14.0
    Episode 487: steps = 13 , reward = -13.0
    Episode 488: steps = 22 , reward = -121.0
    Episode 489: steps = 16 , reward = -115.0
    Episode 490: steps = 15 , reward = -15.0
    Episode 491: steps = 17 , reward = -116.0
    Episode 492: steps = 13 , reward = -13.0
    Episode 493: steps = 25 , reward = -223.0
    Episode 494: steps = 15 , reward = -15.0
    Episode 495: steps = 13 , reward = -13.0
    Episode 496: steps = 29 , reward = -29.0
    Episode 497: steps = 29 , reward = -227.0
    Episode 498: steps = 17 , reward = -17.0
    Episode 499: steps = 16 , reward = -16.0
    test reward = -13.0

