import numpy as np
import random

# 五子棋游戏环境
class wuziqi():
    def __init__(self, board_size=15):
        self.board_size = board_size  # 棋盘盘面大小
        self.board_num = board_size * board_size  # 棋盘所有位置数量

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)  # 棋盘，0代表空位，1代表白棋，-1代表黑棋
        self.side = 1  # 当前下棋的一方，1代表黑方，0代表白方
        self.player_first = False
        self.reward = 0
        self.done = False
        self.last_action = -1

    def reset(self):  # 重置棋盘
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)  # 棋盘，0代表空位，1代表白棋，-1代表黑棋
        self.side = 1  # 当前下棋的一方，1代表黑方，1代表白方
        self.player_first = True if random.random() >= 0.5 else False  # 随机选择谁先下
        self.reward = 0
        self.done = False
        self.last_action = -1
        return self.get_state()
    
    def get_state(self):
        '''
        第0层代表黑子位置
        第1层代表白子位置
        第2层代表上一次落子位置
        第3层全1代表该黑棋下，全0代表该白棋下
        '''
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        state[0, self.board == 1] = 1
        state[1, self.board == -1] = 1
        if self.last_action != -1:
            row, col = self.action2rowcol(self.last_action)
            state[2, row, col] = 1
        state[3] = self.side
        return state

    def step(self, action):  # 下棋
        # 检测当前下棋的位置能不能下
        row, col = self.action2rowcol(action)
        if not self.check_space(row, col):  # 如果把棋下在了本来就有棋的位置
            reward = -2
            info = 'error'
        elif self.done:
            reward = self.reward
            info = 'done'
        else:
            self.board[row][col] = 1 if self.side == 1 else -1  # 下棋
            reward, info = self.judge(row, col)  # 判断输赢
            self.last_action = action
        self.change_side()
        # 分情况判断
        if reward == 0:  # 继续游戏
            pass
        elif reward == 1 or reward == -1:  # 分出胜负
            self.reward = reward
            self.done = True
        elif reward == 2:  # 没有空位继续下棋且没分出胜负
            self.reward = 0
            self.done = True
        else:  # 或下棋出错
            self.reward = reward
            self.done = True
        return self.get_state(), self.reward, self.done, info

    def judge(self, row, col):  # 判断输赢，返回reward奖励
        '''
        reward奖励解释：
        0表示没分出胜负，继续下棋
        1表示黑方胜利
        -1表示白方胜利
        2表示棋盘上没有空位，没法继续下棋，没分出胜负
        -2表示因为不符合规范，下棋出错
        '''

        # 判断横向
        for y in range(self.board_size - 4):
            s = np.sum(self.board[row, y:y + 5])
            if s == 5:
                return 1, [[row, y + i] for i in range(5)]
            if s == -5:
                return -1, [[row, y + i] for i in range(5)]
        
        # 判断纵向
        for x in range(self.board_size - 4):
            s = np.sum(self.board[x:x + 5, col])
            if s == 5:
                return 1, [[x + i, col] for i in range(5)]
            if s == -5:
                return -1, [[x + i, col] for i in range(5)]
        
        # 判断左上到右下
        a = min(row, col)
        r, c = row - a, col - a
        while r + 5 <= self.board_size and c + 5 <= self.board_size:  # 判断斜向是否至少有5个子
            s = 0
            for i in range(5):
                s += self.board[r + i][c + i]
            if s == 5:
                return 1, [[r + i, c + i] for i in range(5)]
            if s == -5:
                return -1, [[r + i, c + i] for i in range(5)]
            r += 1
            c += 1
        
        # 判断右上到左下
        a = min(row, self.board_size - col - 1)
        r, c = row - a, col + a
        while r + 5 <= self.board_size and c - 4 >= 0:  # 判断斜向是否至少有5个子
            s = 0
            for i in range(5):
                s += self.board[r + i][c - i]
            if s == 5:
                return 1, [[r + i, c - i] for i in range(5)]
            if s == -5:
                return -1, [[r + i, c - i] for i in range(5)]
            r += 1
            c -= 1

        # 谁也没赢，检查棋盘上还有没有空位
        if len(np.where(self.board == 0)[0]) == 0:  # 棋盘上没有空位，没法继续下棋，没分出胜负
            return 2, 'no space'
        else:  # 棋盘上还有空位，可以继续下棋
            return 0, 'continue'

    def check_space(self, row, col):
        # 判断是否有棋
        if self.board[row][col] == 0:
            return True
        else:
            return False
    
    def get_space(self):
        return np.where(self.board.reshape(-1) == 0)[0]
    
    def random_action(self, space_only=True):
        if space_only:
            return random.choice(self.get_space())
        else:
            return random.randint(0, self.board_num - 1)
    
    def change_side(self):
        self.side = 0 if self.side == 1 else 1
    
    def rowcol2action(self, row, col):
        return row * self.board_size + col
    
    def action2rowcol(self, action):
        return action // self.board_size, action % self.board_size

# env = wuziqi(6)
# env.reset()
# # print(env.player_first)
# while True:
#     print(env.board)
#     action = int(input())
#     _, reward, done, info = env.step(action)
#     if done:
#         print(reward, info)
#         break