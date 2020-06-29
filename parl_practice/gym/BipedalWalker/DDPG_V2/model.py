# -*- coding: utf-8 -*-

import paddle.fluid as fluid
import parl
from parl import layers
class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        hid3_size = 128
        self.fc1 = layers.fc(size=hid1_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(size=hid2_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc3 = layers.fc(size=hid3_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc4 = layers.fc(size=act_dim, act='tanh', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def policy(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class CriticModel(parl.Model):
    def __init__(self):

        hid1_size = 128
        hid2_size = 128

        self.fc1 = layers.fc(size=hid1_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(size=hid2_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)
        hid1 = self.fc1(obs)
        concat = layers.concat([hid1, act], axis=1)
        hid2 = self.fc2(concat)
        Q = self.fc3(hid2)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class BipedalWalkerModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()