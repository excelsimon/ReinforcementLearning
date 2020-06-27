#coding:utf-8
from ple.games.snake import Snake
from ple import PLE
import numpy as np
def get_obs(env):
    obs = env.getScreenGrayscale() / 255.0
    return obs.astype(np.float).ravel()
    # obs = []
    # game_state = env.getGameState()
    # """
    # {'snake_head_x': 32.0, 'snake_head_y': 32.0, 'food_x': 24, 'food_y': 30,
    # 'snake_body': [0.0, 3.0, 6.0],
    #  'snake_body_pos': [[32.0, 32.0], [29.0, 32.0], [26.0, 32.0]]}
    #
    # {'snake_head_x': 33.93333333333334, 'snake_head_y': 32.0, 'food_x': 36, 'food_y': 30,
    # 'snake_body': [0.0, 1.0933333333333337, 2.5333333333333314, 5.233333333333334],
    # 'snake_body_pos': [[33.93333333333334, 32.0], [32.84, 32.0], [31.400000000000006, 32.0], [28.700000000000003, 32.0]]}
    # """
    # obs.append(game_state['snake_head_x'])
    # obs.append(game_state['snake_head_y'])
    # obs.append(game_state['food_x'])
    # obs.append(game_state['food_y'])
    # body_positions = []
    # obs.append(np.mean(np.array(game_state['snake_body'])))
    # for body_pos in game_state['snake_body_pos']:
    #     body_positions.extend(body_pos)
    # obs.append(np.mean(np.array(body_positions)))
    return obs

if __name__ == '__main__':
    game = Snake(width=64, height=64, init_length=20)
    p = PLE(game, fps=30, display_screen=True, force_fps=True)
    # 根据parl框架构建agent
    print(p.getActionSet())

    act_dim = len(p.getActionSet())
    game_state = p.getGameState()
    print(game_state)