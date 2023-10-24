import torch
from torch import nn
from torch.nn import functional as F

import time
import copy
import random
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

import sys
import os
sys.path.append('C:/python/wuziqi/alphazero1')
os.chdir('C:/python/wuziqi/alphazero1')

from environment import *
from mcts import *
import model_pytorch

def collect_data(env, mcts_alphazero, model, device):
    # 经验缓存 experience buffer
    state_list = []
    prob_list = []
    # 新一轮游戏 new game
    state = env.reset()
    mcts_alphazero.update_with_move(-1)
    # 正式开始 start
    while True:
        action, prob = mcts_alphazero.get_action_prob(env, model, device, 1.0, True)
        next_state, reward, done, info = env.step(action)
        mcts_alphazero.update_with_move(action)
        state_list.append(state)
        prob_list.append(prob)
        state = next_state
        if done:
            reward_list = np.zeros(len(state_list), np.float32)
            reward_list[::2] = reward
            reward_list[1::2] = -reward
            break
    return state_list, prob_list, reward_list

def collect_data_mp(env, mcts_alphazero, model, device, queue, n):
    # 复制环境 copy environment
    env_process = copy.deepcopy(env)
    mcts_process = copy.deepcopy(mcts_alphazero)
    model.to(device)
    # 计数 count
    i = 0
    while True:
        # 经验缓存 experience buffer
        state_list = []
        prob_list = []
        # 新一轮游戏 new game
        state = env_process.reset()
        mcts_process.update_with_move(-1)
        # 正式开始 start
        while True:
            action, prob = mcts_process.get_action_prob(env_process, model, device, 1.0, True)
            next_state, reward, done, info = env_process.step(action)
            mcts_process.update_with_move(action)
            state_list.append(state)
            prob_list.append(prob)
            state = next_state
            if done:
                reward_list = np.zeros(len(state_list), np.float32)
                reward_list[::2] = reward
                reward_list[1::2] = -reward
                break
        # 把数据放到队列 put data into queue
        while True:
            if queue.qsize() >= 10:
                time.sleep(0.2)
                continue
            else:
                queue.put([state_list, prob_list, reward_list])
                break
        # 计数 count
        i += 1
        if i == n:
            break

if __name__ == '__main__':
    board_size = 8
    env = wuziqi(board_size)

    # 主网络
    model = model_pytorch.Net()
    mcts_alphazero = MCTS_AlphaZero(5, 400)

    # 开始生成数据 start generating data
    print('开始生成数据 Start generating data')

    # cpu
    device = torch.device('cpu')

    # cpu单线程
    model.to(device)
    t = time.time()
    num_game = 12
    count = 0
    for i in range(num_game):
        state_list, prob_list, reward_list = collect_data(env, mcts_alphazero, model, device)
        count += len(state_list)
    duration = time.time() - t
    print(f'cpu单线程  duration: {duration:.4f}秒  count: {count}次  speed: {count / duration:.4f}次/秒')

    # cpu双线程
    model.to(device)
    t = time.time()
    queue = mp.Queue()
    process_num = 2
    num_game = 12
    count = 0
    num_game_process = num_game // process_num
    num_game = num_game_process * process_num
    model.cpu()
    process_list = [mp.Process(target=collect_data_mp, args=(env, mcts_alphazero, model, device, queue, num_game_process)) for _ in range(process_num)]
    for i in range(process_num):
        process_list[i].start()
    for i in range(num_game):
        state_list, prob_list, reward_list = queue.get()
        count += len(state_list)
    model.to(device)
    duration = time.time() - t
    print(f'cpu双线程  duration: {duration:.4f}秒  count: {count}次  speed: {count / duration:.4f}次/秒')

    # gpu
    device = torch.device('cuda')

    # gpu单线程
    model.to(device)
    t = time.time()
    num_game = 12
    count = 0
    for i in range(num_game):
        state_list, prob_list, reward_list = collect_data(env, mcts_alphazero, model, device)
        count += len(state_list)
    duration = time.time() - t
    print(f'gpu单线程  duration: {duration:.4f}秒  count: {count}次  speed: {count / duration:.4f}次/秒')

    # gpu双线程
    model.to(device)
    t = time.time()
    queue = mp.Queue()
    process_num = 2
    num_game = 12
    count = 0
    num_game_process = num_game // process_num
    num_game = num_game_process * process_num
    model.cpu()
    process_list = [mp.Process(target=collect_data_mp, args=(env, mcts_alphazero, model, device, queue, num_game_process)) for _ in range(process_num)]
    for i in range(process_num):
        process_list[i].start()
    for i in range(num_game):
        state_list, prob_list, reward_list = queue.get()
        count += len(state_list)
    model.to(device)
    duration = time.time() - t
    print(f'gpu双线程  duration: {duration:.4f}秒  count: {count}次  speed: {count / duration:.4f}次/秒')

    # gpu四线程
    device = torch.device('cuda')
    model.to(device)
    t = time.time()
    queue = mp.Queue()
    process_num = 4
    num_game = 12
    count = 0
    num_game_process = num_game // process_num
    num_game = num_game_process * process_num
    model.cpu()
    process_list = [mp.Process(target=collect_data_mp, args=(env, mcts_alphazero, model, device, queue, num_game_process)) for _ in range(process_num)]
    for i in range(process_num):
        process_list[i].start()
    for i in range(num_game):
        state_list, prob_list, reward_list = queue.get()
        count += len(state_list)
    model.to(device)
    duration = time.time() - t
    print(f'gpu四线程  duration: {duration:.4f}秒  count: {count}次  speed: {count / duration:.4f}次/秒')

    # # gpu六线程
    # device = torch.device('cuda')
    # model.to(device)
    # t = time.time()
    # queue = mp.Queue()
    # process_num = 6
    # num_game = 12
    # count = 0
    # num_game_process = num_game // process_num
    # num_game = num_game_process * process_num
    # model.cpu()
    # process_list = [mp.Process(target=collect_data_mp, args=(env, mcts_alphazero, model, device, queue, num_game_process)) for _ in range(process_num)]
    # for i in range(process_num):
    #     process_list[i].start()
    # for i in range(num_game):
    #     state_list, prob_list, reward_list = queue.get()
    #     count += len(state_list)
    # model.to(device)
    # duration = time.time() - t
    # print(f'gpu六线程  duration: {duration:.4f}秒  count: {count}次  speed: {count / duration:.4f}次/秒')