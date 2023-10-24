import torch
from torch import nn
from torch.nn import functional as F

import copy
import random
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append('C:/python')
sys.path.append('C:/python/wuziqi/alphazero1')
os.chdir('C:/python/wuziqi/alphazero1')

from general_function import *

from environment import *
from mcts import *
from model_pytorch import *

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

def collect_data_mp(env, mcts_alphazero, model, device, game_num, queue):
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
            next_state, reward, done, _ = env_process.step(action)
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
        queue.put([state_list, prob_list, reward_list])
        # 计数 count
        i += 1
        if i == game_num:
            break

def evaluate_model(env, mcts_alphazero, model, device, mcts_pure, game_num=12):
    model.to(device)
    win, lose, tie = 0, 0, 0
    for _ in range(game_num):
        # 新游戏 new game
        env.reset()
        mcts_alphazero.update_with_move(-1)
        # 对手先下 opponent first
        if not env.player_first:
            mcts_pure.update_with_move(-1)
            action = mcts_pure.get_action(env)
            env.step(action)
            mcts_alphazero.update_with_move(action)
        
        # 正式开始 start
        while True:
            # 自己下
            action, _ = mcts_alphazero.get_action_prob(env, model, device)
            _, reward, done, _ = env.step(action)
            mcts_alphazero.update_with_move(action)
            if done:
                break

            # 对手下
            mcts_pure.update_with_move(-1)
            action = mcts_pure.get_action(env)
            _, reward, done, _ = env.step(action)
            mcts_alphazero.update_with_move(action)
            if done:
                break
        
        # 统计输赢 count result
        if reward == 0:
            tie += 1
        elif (reward == 1 and env.player_first) or (reward == -1 and not env.player_first):
            win += 1
        else:
            lose += 1
        
    return win, lose, tie

def evaluate_model_mp(env, mcts_alphazero, model, device, mcts_pure, game_num, queue):
    env_process = copy.deepcopy(env)
    mcts_alphazero_process = copy.deepcopy(mcts_alphazero)
    mcts_pure_process = copy.deepcopy(mcts_pure)
    win, lose, tie = evaluate_model(env_process, mcts_alphazero_process, model, device, mcts_pure_process, game_num)
    queue.put([win, lose, tie])

if __name__ == '__main__':
    board_size = 15
    num_channel = 4
    batch_size = 512
    memory_size = 20000
    num_data_start = batch_size * 4
    learning_rate = 0.001
    learning_rate_multiplier = 1.0
    l2_penalty = 0.0001
    kl_target = 0.02

    num_epoch = 20000
    save_num = 40
    process_num = 3

    # 训练时每个epoch玩多少次游戏 how many games to play while training in every epoch
    game_num = 6
    game_num_process = game_num // process_num
    game_num = game_num_process * process_num
    # 测试时玩多少次游戏 how many games to play while testing
    test_game_num = 12
    test_game_num_process = test_game_num // process_num
    test_game_num = test_game_num_process * process_num

    # device = torch.device('cuda')
    device = training_device(True)
    env = wuziqi(board_size)

    # 创建模型文件夹 create model directory
    create_dir('./model')

    class Agent():
        def __init__(self, model, batch_size=256, memory_size=10000):
            # 基础参数 basic parameters
            self.model = model
            self.batch_size = batch_size
            self.memory_size = memory_size
            # 损失函数和优化器 loss function and optimizer
            self.mse_loss_fn = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
            # 储存经验回放 save experience replay
            self.state_memory = np.zeros((self.memory_size, num_channel, board_size, board_size), dtype=np.float32)
            self.prob_memory = np.empty((self.memory_size, board_size * board_size), dtype=np.float32)
            self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
            self.pointer = 0

        def save_data_list(self, state_list, prob_list, reward_list):
            for state, prob, reward in zip(state_list, prob_list, reward_list):
                agent.save_data(state, prob, reward)

        def save_data(self, state, prob, reward):  # 保存经验回放数据
            for i in range(4):
                # 逆时针旋转90度 rotate 90 degrees counterclockwise
                equi_state = np.rot90(state, i, (1, 2))
                equi_prob = np.rot90(prob.reshape(board_size, board_size), i)
                self.save_data_single(equi_state, equi_prob.reshape(-1), reward)
                # 水平翻转 flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_prob = np.fliplr(equi_prob)
                self.save_data_single(equi_state, equi_prob.reshape(-1), reward)

        def save_data_single(self, state, prob, reward):  # 保存经验回放数据
            index = self.pointer % self.memory_size
            self.state_memory[index] = state
            self.prob_memory[index] = prob
            self.reward_memory[index] = reward
            self.pointer += 1
        
        def sample_data(self):
            # 随机选取训练数据 randomly choose training data
            rand_index = np.random.choice(min(self.memory_size, self.pointer), self.batch_size, replace=False)
            batch_state = torch.tensor(self.state_memory[rand_index], dtype=torch.float32).to(device)
            batch_prob = torch.tensor(self.prob_memory[rand_index], dtype=torch.float32).to(device)
            batch_reward = torch.tensor(self.reward_memory[rand_index], dtype=torch.float32).to(device)
            return batch_state, batch_prob, batch_reward

        def learn(self, batch_state=None, batch_prob=None, batch_reward=None):
            if (batch_state is None) or (not batch_prob is None) or (not batch_reward is None):
                batch_state, batch_prob, batch_reward = self.sample_data()
            # 计算概率和价值 calculate probability and value
            log_prob, value = self.model(batch_state)
            # 计算损失 calculate loss
            value_loss = F.mse_loss(value.view(-1), batch_reward)
            policy_loss = -torch.mean(torch.sum(batch_prob * log_prob, 1))
            loss = value_loss + policy_loss
            set_learning_rate(self.optimizer, learning_rate * learning_rate_multiplier)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return value_loss.item(), policy_loss.item()

    # 主网络
    model = Net().to(device)
    agent = Agent(model, batch_size, memory_size)
    mcts_pure = MCTS_Pure(n_playout=1000)
    mcts_alphazero = MCTS_AlphaZero(5, 600)
    epoch, _ = read_model('./model/model.pt', agent.model, agent.optimizer)

    # 开始生成数据 start generating data
    print('开始生成数据 Start generating data')
    queue = mp.Queue()
    agent.model.cpu()
    process_list = [mp.Process(target=collect_data_mp, args=(env, mcts_alphazero, agent.model, device, -1, queue)) for _ in range(process_num)]
    for i in range(process_num):
        process_list[i].start()
    while True:
        state_list, prob_list, reward_list = queue.get()
        agent.save_data_list(state_list, prob_list, reward_list)
        # 打印进度 print process
        print(f'\r已完成：{(agent.pointer / num_data_start * 100.0):.2f} % / 100.00 %', end='')
        # 如果超过预定次数，就结束游戏 when it passes the defined times, stop the game
        if agent.pointer >= num_data_start:
            break
    for process in process_list:
        process.kill()
    agent.model.to(device)
    print('\n')
    
    # 开始学习 start learning
    while epoch <= num_epoch:
        queue = mp.Queue()
        agent.model.cpu()
        process_list = [mp.Process(target=collect_data_mp, args=(env, mcts_alphazero, agent.model, device, game_num_process, queue)) for _ in range(process_num)]
        for i in range(process_num):
            process_list[i].start()
        for i in range(game_num):
            state_list, prob_list, reward_list = queue.get()
            agent.save_data_list(state_list, prob_list, reward_list)
        agent.model.to(device)
        
        # 更新网络参数 update model parameters
        # num_step = 2
        # num_learn = 5
        # for i in range(num_step):
        #     value_loss_list, policy_loss_list = [], []
        #     batch_state, batch_prob, batch_reward = agent.sample_data()
        #     old_prob, _ = agent.model(batch_state)
        #     for j in range(num_learn):
        #         value_loss, policy_loss = agent.learn(batch_state, batch_prob, batch_reward)
        #         value_loss_list.append(value_loss)
        #         policy_loss_list.append(policy_loss)
        #     new_prob, _ = agent.model(batch_state)
        #     # 计算KL散度，根据KL散度调整学习率倍率 dynamic learning rate based on KL divergence
        #     kl = torch.mean(torch.sum(torch.exp(old_prob) * (old_prob - new_prob), dim=1)).item()
        #     if kl > kl_target * 2:
        #         learning_rate_multiplier = max(learning_rate_multiplier / 1.5, 0.1)
        #     elif kl < kl_target / 2:
        #         learning_rate_multiplier = min(learning_rate_multiplier * 1.5, 10)
        #     print(f'epoch: {epoch}/{num_epoch}  '
        #           f'step: {i+1}/{num_step}  '
        #           f'value_loss: {sum(value_loss_list)/num_learn:.6f}  '
        #           f'policy_loss: {sum(policy_loss_list)/num_learn:.6f}  '
        #           f'kl: {kl:.6f}  '
        #           f'lr_multiplier: {learning_rate_multiplier:.4f}'
        #     )
        
        num_step = 6
        num_learn = 5
        for i in range(num_step):
            value_loss_list, policy_loss_list = [], []
            batch_state, batch_prob, batch_reward = agent.sample_data()
            with torch.no_grad():
                old_prob, _ = agent.model(batch_state)
            for j in range(num_learn):
                value_loss, policy_loss = agent.learn()
                value_loss_list.append(value_loss)
                policy_loss_list.append(policy_loss)
            with torch.no_grad():
                new_prob, _ = agent.model(batch_state)
            # 计算KL散度，根据KL散度调整学习率倍率 dynamic learning rate based on KL divergence
            kl = torch.mean(torch.sum(torch.exp(old_prob) * (old_prob - new_prob), dim=1)).item()
            if kl > kl_target * 2:
                learning_rate_multiplier = max(learning_rate_multiplier / 1.5, 0.1)
            elif kl < kl_target / 2:
                learning_rate_multiplier = min(learning_rate_multiplier * 1.5, 10)
            print(f'epoch: {epoch}/{num_epoch}  '
                  f'step: {i+1}/{num_step}  '
                  f'value_loss: {sum(value_loss_list)/num_learn:.6f}  '
                  f'policy_loss: {sum(policy_loss_list)/num_learn:.6f}  '
                  f'kl: {kl:.6f}  '
                  f'lr_multiplier: {learning_rate_multiplier:.4f}'
            )

        if epoch % save_num == 0:
            # 保存模型 save model
            save_model(f'model/model.pt', agent.model, agent.optimizer, epoch)
            
            # 测试模型 test model
            queue = mp.Queue()
            win, lose, tie = 0, 0, 0
            agent.model.cpu()
            process_list = [mp.Process(target=evaluate_model_mp, args=(env, mcts_alphazero, agent.model, device, mcts_pure, test_game_num_process, queue)) for _ in range(process_num)]
            for i in range(process_num):
                process_list[i].start()
            for i in range(process_num):
                w, l, t = queue.get()
                win += w
                lose += l
                tie += t
            agent.model.to(device)
            
            print(f'Play with MCTS_Pure_{mcts_pure.n_playout}  win: {win}  lose: {lose}  tie: {tie}')

            # 提升难度 raise difficulty
            if lose == 0:
                mcts_pure.n_playout = min(mcts_pure.n_playout + 1000, 5000)

        epoch += 1
