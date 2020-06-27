#coding:utf-8
from ple.games.pong import Pong
from ple import PLE
import numpy as np
def get_obs(env):
    # game_state = env.getGameState()
    # obs = list(game_state.values())
    obs = env.getScreenGrayscale()/255.0
    return obs.astype(np.float).ravel()

if __name__ == '__main__':
    game = Pong(width=128, height=96,MAX_SCORE=11)
    p = PLE(game, fps=30, display_screen=True, force_fps=True)
    # 根据parl框架构建agent
    print(p.getActionSet())
    act_dim = len(p.getActionSet())
    p.getScreenGrayscale()
    game_state = p.getGameState()
    print(game_state)