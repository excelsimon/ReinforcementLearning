#coding:utf-8
from ple.games.pong import Pong
from ple import PLE
import numpy as np
def get_obs(env):
    # game_state = env.getGameState()
    # obs = list(game_state.values())
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    # image = env.getScreenRGB()
    # image = image[35:195]  # 裁剪
    # image = image[::2, ::2, 0]  # 下采样，缩放2倍
    # image[image == 144] = 0  # 擦除背景 (background type 1)
    # image[image == 109] = 0  # 擦除背景 (background type 2)
    # image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    obs = env.getScreenGrayscale() / 255.0
    return obs.astype(np.float).ravel()


if __name__ == '__main__':
    game = Pong(width=128, height=128,MAX_SCORE=11)
    p = PLE(game, fps=30, display_screen=True, force_fps=True)
    # 根据parl框架构建agent
    print(p.getActionSet())
    #obs = p.getScreenRGB()
    obs = p.getScreenGrayscale()
    print(obs)
    print(obs.shape)

    act_dim = len(p.getActionSet())
    game_state = p.getGameState()
    print(game_state)