# -*- coding: utf-8 -*-

import paddle.fluid as fluid
import parl
from parl import layers
import numpy as np

class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hidden_dim_1, hidden_dim_2 = 64, 32
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act='softmax')

    def policy(self, obs):
        x = self.fc1(obs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticModel(parl.Model):
    def __init__(self):
        hidden_dim_1, hidden_dim_2 = 64, 32
        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        x = self.fc1(obs)
        concat = layers.concat([x, act], axis=1)
        x = self.fc2(concat)
        Q = self.fc3(x)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class PongModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()
