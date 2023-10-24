import wx
from enum import Enum
import torch
import numpy as np

import sys
import os
sys.path.append('C:/python')
sys.path.append('C:/python/wuziqi/alphazero1')
os.chdir('C:/python/wuziqi/alphazero1')

from general_function import *

from environment import *
from mcts import *
import model_pytorch
import condition

class player(Enum):
    self = 0
    pvcb = 1
    pvcw = 2
    pvab = 3
    pvaw = 4

class MyFrame(wx.Frame):
    def __init__(self):
        # game environment
        self.board_size = 8
        self.env = wuziqi(self.board_size)

        self.title = 'Renju Program'
        width_compensate = 16
        heigth_compensate = 39
        self.width = 1500 + width_compensate
        self.height = 900 + heigth_compensate

        self.board_interval = 50
        self.board_x1 = 700 + ((700 - (self.board_size - 1) * self.board_interval) // 2)
        self.board_y1 = 100 + ((700 - (self.board_size - 1) * self.board_interval) // 2)
        self.board_x2 = self.board_x1 + (self.board_size - 1) * self.board_interval
        self.board_y2 = self.board_y1 + (self.board_size - 1) * self.board_interval
        self.piece_radius = 20

        self.text_title = 'Renju Program'
        self.text_designer = 'Designed by Chengling Xu'
        self.text_hint = 'Please choose to play with:'
        self.text_rb1 = 'Self'
        self.text_rb2 = 'Condition: black'
        self.text_rb3 = 'Condition: white'
        self.text_rb4 = 'AI: black'
        self.text_rb5 = 'AI: white'
        self.text_new_game = 'New Game'
        self.text_information = 'Python GUI with wxPython'

        self.text_game_finish = 'game finished'
        self.text_black_win = 'Congratulation! Black wins.'
        self.text_white_win = 'Congratulation! White wins.'
        self.text_nobody_win = 'There is no more space on the board. Nobody wins.'
        self.board_text = [''] * self.env.board_num

        self.bg_color = wx.Colour(255, 200, 150)
        self.win_line_color = wx.Colour(180, 180, 255)

        self.board_image_path = 'wood.jpg'

        # initialize frame
        super(MyFrame, self).__init__(None, id=-1, title=self.title, size=(self.width, self.height), style=wx.MINIMIZE_BOX|wx.CAPTION|wx.CLOSE_BOX)
        self.Bind(wx.EVT_LEFT_UP, self.OnBoardClick)
        self.Bind(wx.EVT_MOVE_END, self.OnPaint)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        # build a panel
        self.panel = wx.Panel(self, -1, pos=(0,0), size=(0, 0))

        # build a button
        self.button = wx.Button(self, id=-1, label=self.text_new_game, pos=(200, 750), size=(200, 50))
        self.font_button = wx.Font(20, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.button.SetFont(self.font_button)
        self.button.Bind(wx.EVT_BUTTON, self.OnNewGameClick)

        # build 5 radio buttons
        self.rb1 = wx.RadioButton(self, -1, label=self.text_rb1, pos=(80, 450), size=(300, 50))
        self.rb2 = wx.RadioButton(self, -1, label=self.text_rb2, pos=(80, 500), size=(300, 50))
        self.rb3 = wx.RadioButton(self, -1, label=self.text_rb3, pos=(80, 550), size=(300, 50))
        self.rb4 = wx.RadioButton(self, -1, label=self.text_rb4, pos=(80, 600), size=(300, 50))
        self.rb5 = wx.RadioButton(self, -1, label=self.text_rb5, pos=(80, 650), size=(300, 50))
        self.font_rb = wx.Font(22, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.rb1.SetFont(self.font_rb)
        self.rb2.SetFont(self.font_rb)
        self.rb3.SetFont(self.font_rb)
        self.rb4.SetFont(self.font_rb)
        self.rb5.SetFont(self.font_rb)
        self.rb1.SetBackgroundColour(self.bg_color)
        self.rb2.SetBackgroundColour(self.bg_color)
        self.rb3.SetBackgroundColour(self.bg_color)
        self.rb4.SetBackgroundColour(self.bg_color)
        self.rb5.SetBackgroundColour(self.bg_color)
        self.rb1.SetValue(True)

        # show window
        self.Show(True)
        self.Center()

        self.dc = wx.ClientDC(self)

        # 让ai和mcts_pure对战 let ai and mcts_pure fight with each other
        # self.timer = wx.Timer(self)
        # self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)

        self.enabled = False
        self.opponent = player.self

        self.mcts_alphazero = MCTS_AlphaZero(5, 600)
        self.mcts_pure = MCTS_Pure(5, 1000)

        self.device = torch.device('cuda')

        self.model = model_pytorch.Net2().to(self.device)
        read_model('model2/model.pt', self.model)

    def reset(self):
        self.enabled = False
        self.env.reset()
        self.board_text = [''] * self.env.board_num
        self.draw_base()
    
    def draw_base(self):
        # set background color
        self.dc.SetBackground(wx.Brush(self.bg_color))
        self.dc.Clear()
        # draw text and board background
        self.draw_text()
        self.draw_board()
    
    def draw_text(self):
        self.dc.SetTextForeground('black')

        # draw title text
        font_title = wx.Font(56, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.dc.SetFont(font_title)
        self.dc.DrawText(self.text_title, 50, 100)

        # draw designer text
        font_desiner = wx.Font(18, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.dc.SetFont(font_desiner)
        self.dc.DrawText(self.text_designer, 320, 250)

        # draw text - Please choose to play with:
        font_hint = wx.Font(24, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.dc.SetFont(font_hint)
        self.dc.DrawText(self.text_hint, 80, 400)

        # draw information text
        font_information = wx.Font(18, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.dc.SetFont(font_information)
        self.dc.DrawText(self.text_information, 50, 850)

    def draw_board(self):
        # draw background
        self.dc.DrawBitmap(wx.Bitmap(self.board_image_path), 650, 50)

        # draw lines
        self.dc.SetPen(wx.Pen('black', width=1, style=wx.SOLID))
        for i in range(self.board_size):
            # horizontal lines
            self.dc.DrawLine(self.board_x1, self.board_y1 + self.board_interval * i, self.board_x2, self.board_y1 + self.board_interval * i)
            # vertical lines
            self.dc.DrawLine(self.board_x1 + self.board_interval * i, self.board_y1, self.board_x1 + self.board_interval * i, self.board_y2)
        
        # draw the dots on the board
        if self.board_size == 15:
            self.dc.SetBrush(wx.Brush('black'))
            self.dc.DrawCircle(850, 250, 3)
            self.dc.DrawCircle(1250, 250, 3)
            self.dc.DrawCircle(850, 650, 3)
            self.dc.DrawCircle(1250, 650, 3)
            self.dc.DrawCircle(1050, 450, 3)
    
    def draw_piece(self, side, row, col):
        # 计算棋子坐标 calculate the coordinate of the piece
        x = self.board_x1 + self.board_interval * col
        y = self.board_y1 + self.board_interval * row
        if side == 1:  # black
            self.dc.SetBrush(wx.Brush('black'))
        else:  # white
            self.dc.SetBrush(wx.Brush('white'))
        self.dc.DrawCircle(x, y, 20)

    def draw_all_piece(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.env.board[row][col] == 1:
                    self.draw_piece(1, row, col)
                elif self.env.board[row][col] == -1:
                    self.draw_piece(0, row, col)
    
    def draw_win_line(self, info):
        if type(info) == str:
            return
        row1, col1 = info[0]
        row2, col2 = info[4]
        x1, y1 = self.rowcol2xy(row1, col1)
        x2, y2 = self.rowcol2xy(row2, col2)
        self.dc.SetPen(wx.Pen(self.win_line_color, width=6, style=wx.SOLID))
        self.dc.DrawLine(x1, y1, x2, y2)
    
    def show_win_dialog(self):
        if self.env.reward == 1:  # black wins
            wx.MessageBox(self.text_black_win, self.text_game_finish)
        if self.env.reward == -1:  # white wins
            wx.MessageBox(self.text_white_win, self.text_game_finish)
        if self.env.reward == 0:  # nobody wins
            wx.MessageBox(self.text_nobody_win, self.text_game_finish)
    
    def step(self, action):
        # 转换坐标 change coordinate
        row, col = self.env.action2rowcol(action)
        # 检查位置是否有效 check if the action is valid
        if (action != -1) and (self.env.check_space(row, col)):
            # 下棋 step
            self.draw_piece(self.env.side, row, col)
            _, _, done, info = self.env.step(action)

            # 和条件下棋 play with condition
            # if self.opponent == player.pvcb or self.opponent == player.pvcw:
            #     self.mcts_pure.update_with_move(action)
            
            # 和AI下棋 play with AI
            if self.opponent == player.pvab or self.opponent == player.pvaw:
                self.mcts_alphazero.update_with_move(action)
            
            # check win
            if done:
                self.enabled = False
                self.draw_win_line(info)
                self.show_win_dialog()
    
    def draw_board_text(self, text):
        self.dc.SetTextForeground('red')
        font_board_text = wx.Font(10, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.NORMAL, faceName='微软雅黑')
        self.dc.SetFont(font_board_text)
        for row in range(self.board_size):
            for col in range(self.board_size):
                x, y = self.rowcol2xy(row, col)
                self.dc.DrawText(text[self.env.rowcol2action(row, col)], x - 20, y - 10)
    
    def redraw(self):
        self.draw_base()
        self.draw_all_piece()
        self.draw_board_text(self.board_text)
    
    def OnPaint(self, event):
        self.redraw()
    
    def rowcol2xy(self, row, col):
        x = self.board_x1 + self.board_interval * col
        y = self.board_y1 + self.board_interval * row
        return x, y
    
    def xy2rowcol(self, x, y):
        left = self.board_x1 - self.piece_radius
        top = self.board_y1 - self.piece_radius
        if left <= x <= self.board_x2 + self.piece_radius and top <= y <= self.board_y2 + self.piece_radius:
            x_index = (x - left) // self.board_interval
            x_range = (x - left) % self.board_interval
            y_index = (y - top) // self.board_interval
            y_range = (y - top) % self.board_interval
            if (x_range <= self.piece_radius * 2) and (y_range <= self.piece_radius * 2):
                return y_index, x_index
        return -1, -1

    def OnNewGameClick(self, event):
        self.reset()
        
        if self.rb1.GetValue():
            self.opponent = player.self

            self.mcts_pure.update_with_move(-1)
            self.mcts_alphazero.update_with_move(-1)
            action = self.ai_action()
            self.step(action)
            self.mcts_alphazero.update_with_move(action)
            self.timer.Start(1000)

        if self.rb2.GetValue():
            self.opponent = player.pvcb
            self.mcts_pure.update_with_move(-1)

        if self.rb3.GetValue():
            self.opponent = player.pvcw
            self.enabled = False
            self.mcts_pure.update_with_move(-1)
            # action = self.mcts_pure.get_action(self.env)
            action = condition.get_action(self.env, self.env.side, self.board_size)
            self.step(action)
        
        if self.rb4.GetValue():
            self.opponent = player.pvab
            self.mcts_alphazero.update_with_move(-1)

        if self.rb5.GetValue():
            self.opponent = player.pvaw
            self.enabled = False
            self.mcts_alphazero.update_with_move(-1)
            action = self.ai_action()
            self.step(action)

        
        self.enabled = True

    def OnBoardClick(self, event):
        # 坐标转换 change coordinate
        x, y = event.GetPosition()
        row, col = self.xy2rowcol(x, y)
        action = self.env.rowcol2action(row, col)
        # 检查动作是否有效 check if the action is valid
        if row == -1 or not self.enabled or not self.env.check_space(row, col):
            return
        
        # 自己下棋 play with myself
        if (self.opponent == player.self):
            self.step(action)

        # 和条件下棋 play with condition
        if self.opponent == player.pvcb or self.opponent == player.pvcw:
            self.step(action)
            # 下一步 next step
            self.enabled = False
            if not self.env.done:
                # action = self.mcts_pure.get_action(self.env)
                action = condition.get_action(self.env, self.env.side, self.board_size)
                self.step(action)

                if not self.env.done:
                    self.enabled = True

        # 和AI下棋 play with AI
        if self.opponent == player.pvab or self.opponent == player.pvaw:
            self.step(action)
            # 下一步 next step
            self.enabled = False
            if not self.env.done:
                action, action_list, prob, visit_list, Q_list = self.ai_action_prob()
                self.step(action)

                visit = np.zeros(self.env.board_num, np.int32)
                visit[action_list] = visit_list
                self.board_text = [f'{v}' for v in visit]
                self.redraw()

                if not self.env.done:
                    self.enabled = True

    # def OnTimer(self, event):
    #     self.timer.Stop()
    #     if not self.env.done:
    #         if self.env.side == 1:
    #             action = self.ai_action()
    #             self.mcts_alphazero.update_with_move(action)
    #         else:
    #             action = self.mcts_pure_action()
    #             self.mcts_pure.update_with_move(action)
    #         self.step(action)
    #         self.redraw()
    #         self.timer.Start(1000)
    
    def ai_action_prob(self):
        action, action_list, prob, visit_list, Q_list = self.mcts_alphazero.get_action_prob_complete(self.env, self.model, self.device)
        return action, action_list, prob, visit_list, Q_list

    def ai_action(self):
        action, _ = self.mcts_alphazero.get_action_prob(self.env, self.model, self.device)
        return action
    
    def mcts_pure_action(self):
        action = self.mcts_pure.get_action(self.env)
        return action

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    frame.reset()
    app.MainLoop()
