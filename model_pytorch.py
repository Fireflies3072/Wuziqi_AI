import torch
from torch import nn

import sys
sys.path.append('C:/python')
from general_function import *

# class residual_block(nn.Module):
#     def __init__(self, in_dim, middle_dim=0, out_dim=0, normalization=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True),
#                  bias=True, padding_mode='zeros', type='same', structure='basic', pre_activation=False):
#         super(residual_block, self).__init__()
#         # 确定参数
#         middle_dim = in_dim if middle_dim == 0 else middle_dim
#         out_dim = in_dim if out_dim == 0 else out_dim
#         type, structure = type.lower(), structure.lower()
#         # 类型
#         if type in ['same', 'regular', 'conv']:
#             type = 'same'
#         elif type in ['up', 'upsample']:
#             type = 'up'
#         elif type in ['down', 'downsample']:
#             type = 'down'
#         else:
#             raise ValueError("type must be in ['same', 'up', 'down']")
#         # 结构
#         if structure in ['basic', 'bottleneck']:
#             pass
#         elif structure == '33':
#             structure = 'basic'
#         elif (structure == '131' and type == 'same') or (structure == '141' and type in ['up', 'down']):
#             structure = 'bottleneck'
#         else:
#             raise ValueError("structure must be in ['basic', 'bottleneck']")
#         # 添加归一化层
#         def add_normalization(module_list, channel:int):
#             if normalization:
#                 module_list.append(normalization(channel))
        
#         # 主网络
#         layer = []
#         self.after_layer = None
#         if not pre_activation:
#             if structure == 'basic':
#                 # 第一层
#                 layer.append(nn.Conv2d(in_dim, middle_dim, 3, 1, 1, bias=bias, padding_mode=padding_mode))
#                 add_normalization(layer, middle_dim)
#                 layer.append(activation)
#                 # 第二层
#                 if type == 'same':
#                     layer.append(nn.Conv2d(middle_dim, out_dim, 3, 1, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'up':
#                     layer.append(nn.ConvTranspose2d(middle_dim, out_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'down':
#                     layer.append(nn.Conv2d(middle_dim, out_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 add_normalization(layer, out_dim)
#             elif structure == 'bottleneck':
#                 # 第一层
#                 layer.append(nn.Conv2d(in_dim, middle_dim, 1, 1, 0, bias=bias))
#                 add_normalization(layer, middle_dim)
#                 layer.append(activation)
#                 # 第二层
#                 if type == 'same':
#                     layer.append(nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'up':
#                     layer.append(nn.ConvTranspose2d(middle_dim, middle_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'down':
#                     layer.append(nn.Conv2d(middle_dim, middle_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 add_normalization(layer, middle_dim)
#                 layer.append(activation)
#                 # 第三层
#                 layer.append(nn.Conv2d(middle_dim, out_dim, 1, 1, 0, bias=bias))
#                 add_normalization(layer, out_dim)
#             # 相加之后
#             self.after_layer = activation
#         else:  # pre-activation
#             if structure == 'basic':
#                 # 第一层
#                 add_normalization(layer, in_dim)
#                 layer.append(activation)
#                 layer.append(nn.Conv2d(in_dim, middle_dim, 3, 1, 1, bias=bias, padding_mode=padding_mode))
#                 # 第二层
#                 add_normalization(layer, middle_dim)
#                 layer.append(activation)
#                 if type == 'same':
#                     layer.append(nn.Conv2d(middle_dim, out_dim, 3, 1, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'up':
#                     layer.append(nn.ConvTranspose2d(middle_dim, out_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'down':
#                     layer.append(nn.Conv2d(middle_dim, out_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#             elif structure == 'bottleneck':
#                 # 第一层
#                 add_normalization(layer, in_dim)
#                 layer.append(activation)
#                 layer.append(nn.Conv2d(in_dim, middle_dim, 1, 1, 0, bias=bias))
#                 # 第二层
#                 add_normalization(layer, middle_dim)
#                 layer.append(activation)
#                 if type == 'same':
#                     layer.append(nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'up':
#                     layer.append(nn.ConvTranspose2d(middle_dim, middle_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 elif type == 'down':
#                     layer.append(nn.Conv2d(middle_dim, middle_dim, 4, 2, 1, bias=bias, padding_mode=padding_mode))
#                 # 第三层
#                 add_normalization(layer, middle_dim)
#                 layer.append(activation)
#                 layer.append(nn.Conv2d(middle_dim, out_dim, 1, 1, 0, bias=bias))
#             # 相加之后
#             self.after_layer = do_nothing_block()
#         self.conv1 = nn.Sequential(*layer)

#         # 短接网络
#         layer2 = []
#         # 缩放尺寸
#         if type == 'up':
#             layer2.append(nn.UpsamplingNearest2d(scale_factor=2))
#         elif type == 'down':
#             layer2.append(nn.AvgPool2d(2, 2))
#         # 变换维度
#         if in_dim != out_dim:
#             if not pre_activation:
#                 layer2.append(nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias))
#                 add_normalization(layer2, out_dim)
#             else:
#                 add_normalization(layer2, out_dim)
#                 layer2.append(activation)
#                 layer2.append(nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias))
#         if layer2:
#             self.conv2 = nn.Sequential(*layer2)
#         else:
#             self.conv2 = do_nothing_block()

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         return self.after_layer(x1 + x2)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, True),
#             residual_block(64, 64, 64),
#             residual_block(64, 64, 64),
#             residual_block(64, 64, 64),
#             nn.Conv2d(64, 4, 3, 1, 1)
#         )
#         self.policy = nn.Sequential(
#             nn.Linear(900, 256),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(256, 225),
#             nn.LogSoftmax(1)
#         )
#         self.value = nn.Sequential(
#             nn.Linear(900, 64),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(64, 1),
#             nn.Tanh()
#         )
    
#     def forward(self, state):
#         x = self.cnn(state).view(-1, 900)
#         log_prob = self.policy(x)
#         value = self.value(x)
#         return log_prob, value

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(256, 64)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(128, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = torch.relu(self.conv1(state_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # action policy layers
        x_act = torch.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 256)
        x_act = torch.log_softmax(self.act_fc1(x_act), 1)
        # state value layers
        x_val = torch.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 128)
        x_val = torch.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val