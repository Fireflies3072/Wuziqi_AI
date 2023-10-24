import torch
from torch import nn
from torch.nn import functional as F

import copy
import random
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
import imageio

import sys
import os
sys.path.append('C:/python')
sys.path.append('C:/python/wuziqi/alphazero1')
os.chdir('C:/python/wuziqi/alphazero1')

from general_function import *
from environment import *
from mcts import *
from model_pytorch import *

def random_action():
    return env.random_step()

def human_action():
    return int(input())

def mcts_alphazero_action():
    action, _ = mcts_alphazero.get_action_prob(env, model, device)
    return action

def mcts_pure_action():
    action = mcts_pure.get_action(env)
    return action

def update_action(action):
    mcts_alphazero.update_with_move(action)
    mcts_pure.update_with_move(action)

def rowcol2xy(row, col, board_x1, board_y1, board_interval):
    x = board_x1 + board_interval * col
    y = board_y1 + board_interval * row
    return x, y

create_dir('./output')

board_size = 8
num_channel = 4

device = training_device(True)
env = wuziqi(board_size)

# 主网络    
mcts_alphazero = MCTS_AlphaZero(n_playout=400)
mcts_pure = MCTS_Pure(n_playout=2000)

model = Net2().to(device)
epoch, _ = read_model(r'model2/model.pt', model)

player_list = ['AI', 'Pure MCTS']
color_list = [(0, 0, 0), (255, 255, 255)]
board_interval = 50
board_x1 = 50 + (700 - (board_size - 1) * board_interval) // 2
board_y1 = 50 + (700 - (board_size - 1) * board_interval) // 2
board_x2 = board_x1 + (board_size - 1) * board_interval
board_y2 = board_y1 + (board_size - 1) * board_interval
piece_radius = 20

# 测试模型 test model
win, lose, tie = 0, 0, 0
for i in range(10):
    # 新游戏 new game
    action_list = []
    state = env.reset()
    env.player_first = True if i % 2 == 0 else False
    update_action(-1)
    
    # 对手先下 opponent first
    if not env.player_first:
        print(env.board)
        action = mcts_pure_action()
        state, _, _, _ = env.step(action)
        update_action(action)
        action_list.append(action)
    
    # 正式开始 start
    while True:
        # 自己下
        print(env.board)
        action = mcts_alphazero_action()
        next_state, reward, done, info = env.step(action)
        update_action(action)
        action_list.append(action)
        if done:
            break

        # 对手下
        print(env.board)
        action = mcts_pure_action()
        next_state, reward, done, info = env.step(action)
        update_action(action)
        action_list.append(action)
        if done:
            break
    
    print(env.board)
    if reward == 0:
        tie += 1
    elif (reward == 1 and env.player_first) or (reward == -1 and not env.player_first):
        win += 1
    else:
        lose += 1
    
    # 创建动态图
    image_list = []

    frame = read_cv2('wood.jpg')
    cv2.circle(frame, (30, 30), 10, (0, 0, 0), -1)
    cv2.putText(frame, player_list[i % 2], (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.circle(frame, (230, 30), 10, (255, 255, 255), -1)
    cv2.putText(frame, player_list[(i + 1) % 2], (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    for j in range(board_size):
        # horizontal lines
        cv2.line(frame, (board_x1, board_y1 + board_interval * j), (board_x2, board_y1 + board_interval * j), (0, 0, 0), 1)
        # vertical lines
        cv2.line(frame, (board_x1 + board_interval * j, board_y1), (board_x1 + board_interval * j, board_y2), (0, 0, 0), 1)
    image_list.append(frame.copy())
    
    # 画每一步的棋子
    for j in range(len(action_list)):
        row, col = env.action2rowcol(action_list[j])
        x, y = rowcol2xy(row, col, board_x1, board_y1, board_interval)
        cv2.circle(frame, (x, y), piece_radius, color_list[j % 2], -1)
        image_list.append(frame.copy())
    
    # 结束之后
    if env.reward != 0:
        if (env.player_first and env.reward == 1) or (not env.player_first and env.reward == -1):
            winner = 0
        else:
            winner = 1
        row1, col1 = info[0]
        row2, col2 = info[4]
        x1, y1 = rowcol2xy(row1, col1, board_x1, board_y1, board_interval)
        x2, y2 = rowcol2xy(row2, col2, board_x1, board_y1, board_interval)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 30, 255), 5)
        cv2.putText(frame, f'{player_list[winner]} won', (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    else:
        cv2.putText(frame, f'{player_list[0]} and {player_list[1]} tied', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    
    image_list.append(frame.copy())
    image_list.append(frame.copy())

    imageio.mimsave(f'output/game_{i + 1}.gif', image_list, fps=1)


print(win, lose, tie)