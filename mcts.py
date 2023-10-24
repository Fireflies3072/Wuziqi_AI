import copy
import torch
import numpy as np
import math

class TreeNode():
    def __init__(self, parent, P):
        self.parent = parent
        self.children = {}
        self.n_visit = 0
        self.Q = 0
        self.U = 0
        self.P = P
    
    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node:act_node[1].get_value(c_puct))
    
    def expand(self, action_prob):
        for action, prob in action_prob:
            if not action in self.children:
                self.children[action] = TreeNode(self, prob)

    def update(self, leaf_value):
        self.n_visit += 1
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visit

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self.U = c_puct * self.P * math.sqrt(self.parent.n_visit) / (1 + self.n_visit)
        return self.Q + self.U

    def is_leaf(self):
        return self.children == {}
    
    def is_root(self):
        return self.parent is None

class MCTS_Pure():
    def __init__(self, c_puct=5, n_playout=2000):
        self.root = TreeNode(None, 1.0)
        self.c_puct = c_puct
        self.n_playout = n_playout
    
    def playout(self, env):
        # 选择叶节点 choose a leaf node
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            env.step(action)
        # 扩展节点 expand nodes
        if not env.done:
            # 每个空位相同概率 the same probability in every space
            space = env.get_space()
            prob = np.ones(len(space), np.float64) / len(space)
            action_prob = zip(space, prob)
            node.expand(action_prob)
        # 评估并更新节点 evaluate and update node
        leaf_value = self.evaluate_rollout(env)
        node.update_recursive(-leaf_value)
    
    def evaluate_rollout(self, env, limit=1000):
        # 随机下到游戏结束 take random step until the games ends
        side = env.side
        for i in range(limit):
            if env.done:
                break
            action = env.random_step()
            env.step(action)
        # 获胜的人 winner
        winner = -1
        if env.reward == 1:
            winner = 1
        elif env.reward == -1:
            winner = 0
        # 返回评估的值 return the evalution value
        if winner == -1:
            return 0
        else:
            return 1 if side == winner else -1
    
    def get_action(self, env):
        # 复制游戏环境 copy the game environment
        for i in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            self.playout(env_copy)
        # 返回被访问次数最多的位置 return the step with the greatest visit times
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visit)[0]
    
    def update_with_move(self, last_move):
        # 更新根节点 update the root node
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:  # 重置根节点 reset the root node
            self.root = TreeNode(None, 1.0)

class MCTS_AlphaZero():
    def __init__(self, c_puct=5, n_playout=400):
        self.root = TreeNode(None, 1.0)
        self.c_puct = c_puct
        self.n_playout = n_playout
    
    def playout(self, env, model, device):
        # 选择叶节点 choose a leaf node
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            env.step(action)
        # 评估节点 evaluate node
        if not env.done:
            # 扩展节点 expand nodes
            space = env.get_space()
            state = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                log_prob, value = model(state)
            prob = torch.exp(log_prob.view(-1).cpu()).detach().numpy()[space]
            action_prob = zip(space, prob)
            node.expand(action_prob)
            leaf_value = value.view(-1).item()
        else:
            # 获胜的人 winner
            winner = -1
            if env.reward == 1:
                winner = 1
            elif env.reward == -1:
                winner = 0
            # 返回评估的值 return the evalution value
            if winner == -1:
                leaf_value = 0
            else:
                leaf_value = 1 if env.side == winner else -1
        # 更新节点 update node
        node.update_recursive(-leaf_value)
    
    def get_action(self, env, model, device, temp=0.001, explore=False):
        action, _ = self.get_action_prob(env, model, device, temp, explore)
        # 返回选中的动作 return the selected action
        return action
    
    def get_action_prob(self, env, model, device, temp=0.001, explore=False):
        # 复制游戏环境 copy the game environment
        for i in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            self.playout(env_copy, model, device)
        # 空位和访问数 space action and number of visits
        action_visit = [(action, node.n_visit) for action, node in self.root.children.items()]
        action_list, visit_list = zip(*action_visit)
        action_list, visit_list = list(action_list), list(visit_list)
        # 空位的概率 probability of spaces
        prob_space = 1.0 / temp * np.log(np.array(visit_list, np.float32) + 1e-10)
        prob_space = prob_space - np.max(prob_space)
        prob_space = np.exp(prob_space)
        prob_space /= np.sum(prob_space)
        # 所有位置的概率 probability of all positions
        prob = np.zeros(env.board_num, np.float32)
        prob[action_list] = prob_space
        # 增加噪声来探索 add noise to explore
        if explore:
            prob_space = 0.75 * prob_space + 0.25 * np.random.dirichlet(np.full(len(prob_space), 0.3, np.float32))
            prob_space /= prob_space.sum()
        # 随机选择动作 randomly choose action
        action = np.random.choice(action_list, p=prob_space)
        # 返回选中的动作和索引位置的概率 return the selected action and the probability of all positions
        return action, prob
    
    def get_action_prob_complete(self, env, model, device, temp=0.001, explore=False):
        # 复制游戏环境 copy the game environment
        for i in range(self.n_playout):
            env_copy = copy.deepcopy(env)
            self.playout(env_copy, model, device)
        # 空位和访问数 space action and number of visits
        action_visit = [(action, node.n_visit, node.Q) for action, node in self.root.children.items()]
        action_list, visit_list, Q_list = zip(*action_visit)
        action_list, visit_list, Q_list = list(action_list), list(visit_list), list(Q_list)
        # 空位的概率 probability of spaces
        prob_space = 1.0 / temp * np.log(np.array(visit_list, np.float32) + 1e-10)
        prob_space = prob_space - np.max(prob_space)
        prob_space = np.exp(prob_space)
        prob_space /= np.sum(prob_space)
        # 所有位置的概率 probability of all positions
        prob = np.zeros(env.board_num, np.float32)
        prob[action_list] = prob_space
        # 增加噪声来探索 add noise to explore
        if explore:
            prob_space = 0.75 * prob_space + 0.25 * np.random.dirichlet(np.full(len(prob_space), 0.3, np.float32))
            prob_space /= prob_space.sum()
        # 选择动作 choose action
        if explore:
            action = np.random.choice(action_list, p=prob_space)
        else:
            action = action_list[np.argmax(prob_space)]
        # 返回所有信息 return all information
        return action, action_list, prob, visit_list, Q_list
    
    def update_with_move(self, last_move):
        # 更新根节点 update the root node
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:  # 重置根节点 reset the root node
            self.root = TreeNode(None, 1.0)

# mcts = MCTS(policy_value_fn, 5, 3000)
# env = wuziqi(6)
# env.reset()

# if env.player_first:
#     print(env.board)
#     action = int(input())
#     _, reward, done, info = env.step(action)
#     mcts.update_with_move(action)

# while True:
#     print(env.board)
#     # mcts.update_with_move(-1)
#     action = mcts.get_action(env)
#     print(action)
#     _, reward, done, info = env.step(action)
#     mcts.update_with_move(action)
#     if done:
#         print(reward, info)
#         break

#     print(env.board)
#     action = int(input())
#     _, reward, done, info = env.step(action)
#     mcts.update_with_move(action)
#     if done:
#         print(reward, info)
#         break