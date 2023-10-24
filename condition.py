import random
import math

def rowcol2step(row, col, board_size):  # 坐标转换
    return row * board_size + col

def rowcolposdirec2step(row, col, position, board_size, direction):  # 坐标转换
    if direction == 'h':
        return rowcol2step(row, col + position, board_size)
    elif direction == 'v':
        return rowcol2step(row + position, col, board_size)
    elif direction == 'sr':
        return rowcol2step(row + position, col + position, board_size)
    elif direction == 'sl':
        return rowcol2step(row + position, col - position, board_size)

def check_in_board(row, col, board_size):  # 检查一个点是不是在棋盘内
    if (row >= 0) and (row < board_size):
        if (col >= 0) and (col < board_size):
            return True
    return False

def check_line_in_board(row, col, board_size, direction, length):  # 检查一行棋子是不是在棋盘内
    end_row = 0
    end_col = 0
    # 根据方向算出结束点的坐标
    if direction == 'h':
        end_row = row
        end_col = col + length - 1
    elif direction == 'v':
        end_row = row + length - 1
        end_col = col
    elif direction == 'sr':
        end_row = row + length - 1
        end_col = col + length - 1
    elif direction == 'sl':
        end_row = row + length - 1
        end_col = col - length + 1
    # 检查开始点和结束点是不是都在棋盘内
    if check_in_board(row, col, board_size) is True:
        if check_in_board(end_row, end_col, board_size) is True:
            return True
    return False

def check_space(board, row, col, board_size, direction, length):  # 检查一段距离内的空档
    for i in range(length):
        if direction == 'h':  # 横向
            if board[row][col + i] == 0:
                return rowcol2step(row, col + i, board_size)
        elif direction == 'v':  # 竖向
            if board[row + i][col] == 0:
                return rowcol2step(row + i, col, board_size)
        elif direction == 'sr':  # 往右下斜
            if board[row + i][col + i] == 0:
                return rowcol2step(row + i, col + i, board_size)
        elif direction == 'sl':  # 往左下斜
            if board[row + i][col - i] == 0:
                return rowcol2step(row + i, col - i, board_size)
    return -1

def check_space5(board, row, col, board_size, direction):
    return check_space(board, row, col, board_size, direction, 5)

def check_space4(board, row, col, board_size, direction):
    return check_space(board, row, col, board_size, direction, 4)

def check_block_h(board, row, col, board_size, length, distance):  # 检测横向阻挡
    start_block = False
    end_block = False
    col1 = col - distance  # 检测的头
    col2 = col + length + distance - 1  # 检测的尾
    if col1 < 0:  # 如果头边上是棋盘
        start_block = True
    elif board[row][col1] != 0:  # 如果头边上有棋
        start_block = True
    if col2 >= board_size:  # 如果尾边上是棋盘
        end_block = True
    elif board[row][col2] != 0:  # 如果尾边上有棋
        end_block = True
    return start_block, end_block

def check_block_v(board, row, col, board_size, length, distance):  # 检测纵向阻挡
    start_block = False
    end_block = False
    row1 = row - distance  # 检测的头
    row2 = row + length + distance - 1  # 检测的尾
    if row1 < 0:  # 如果头边上是棋盘
        start_block = True
    elif board[row1][col] != 0:  # 如果头边上有棋
        start_block = True
    if row2 >= board_size:  # 如果尾边上是棋盘
        end_block = True
    elif board[row2][col] != 0:  # 如果尾边上有棋
        end_block = True
    return start_block, end_block

def check_block_sr(board, row, col, board_size, length, distance):  # 检测斜右下阻挡
    start_block = False
    end_block = False
    row1 = row - distance  # 检测的头
    col1 = col - distance
    row2 = row + length + distance - 1  # 检测的尾
    col2 = col + length + distance - 1
    if (row1 < 0) or (col1 < 0):  # 如果头边上是棋盘
        start_block = True
    elif board[row1][col1] != 0:  # 如果头边上有棋
        start_block = True
    if (row2 >= board_size) or (col2 >=board_size):  # 如果尾边上是棋盘
        end_block = True
    elif board[row2][col2] != 0:  # 如果尾边上有棋
        end_block = True
    return start_block, end_block

def check_block_sl(board, row, col, board_size, length, distance):  # 检测斜左下阻挡
    start_block = False
    end_block = False
    row1 = row - distance  # 检测的头
    col1 = col + distance
    row2 = row + length + distance - 1  # 检测的尾
    col2 = col - length - distance + 1
    if (row1 < 0) or (col1 >= board_size):  # 如果头边上是棋盘
        start_block = True
    elif board[row1][col1] != 0:  # 如果头边上有棋
        start_block = True
    if (row2 >= board_size) or (col2 < 0):  # 如果尾边上是棋盘
        end_block = True
    elif board[row2][col2] != 0:  # 如果尾边上有棋
        end_block = True
    return start_block, end_block

def check_block(board, row, col, board_size, direction, length, distance):
    if direction == 'h':
        return check_block_h(board, row, col, board_size, length, distance)
    elif direction == 'v':
        return check_block_v(board, row, col, board_size, length, distance)
    elif direction == 'sr':
        return check_block_sr(board, row, col, board_size, length, distance)
    elif direction == 'sl':
        return check_block_sl(board, row, col, board_size, length, distance)

def check_block_next(board, row, col, board_size, direction, length):
    return check_block(board, row, col, board_size, direction, length, 1)

def check_block_one_away(board, row, col, board_size, direction, length):
    return check_block(board, row, col, board_size, direction, length, 2)

def get_line(board, row, col, direction, length):  #返回一行里面的棋子
    line = []
    if direction == 'h':
        for i in range(length):
            line.append(board[row][col + i])
    elif direction == 'v':
        for i in range(length):
            line.append(board[row + i][col])
    elif direction == 'sr':
        for i in range(length):
            line.append(board[row + i][col + i])
    elif direction == 'sl':
        for i in range(length):
            line.append(board[row + i][col - i])
    return line

def get5(board, row, col, direction):  # 返回一行5个棋子
    return get_line(board, row, col, direction, 5)

def get4(board, row, col, direction):  # 返回一行4个棋子
    return get_line(board, row, col, direction, 4)

def get3(board, row, col, direction):  # 返回一行3个棋子
    return get_line(board, row, col, direction, 3)

def get2(board, row, col, direction):  # 返回一行2个棋子
    return get_line(board, row, col, direction, 2)

def get_line_head(row, col, board_size, direction):  # 返回一行棋子的前一个棋子
    return rowcolposdirec2step(row, col, -1, board_size, direction)

def get_line_tail(row, col, board_size, direction, length):  # 返回一行棋子的后一个棋子
    return rowcolposdirec2step(row, col, length, board_size, direction)

def check_win(board, row, col, board_size, direction):  # 检查是不是下一步就可以赢
    line = get5(board, row, col, direction)
    if sum(line) == 4:
        return check_space5(board, row, col, board_size, direction)  # 下在空位的位置
    else:
        return -1

def check_win_all_direction(board, row, col, board_size):
    # 横向
    if check_line_in_board(row, col, board_size, 'h', 5):
        win = check_win(board, row, col, board_size, 'h')
        if win != -1:
            return win
    # 纵向
    if check_line_in_board(row, col, board_size, 'v', 5):
        win = check_win(board, row, col, board_size, 'v')
        if win != -1:
            return win
    # 斜右下
    if check_line_in_board(row, col, board_size, 'sr', 5):
        win = check_win(board, row, col, board_size, 'sr')
        if win != -1:
            return win
    # 斜左下
    if check_line_in_board(row, col, board_size, 'sl', 5):
        win = check_win(board, row, col, board_size, 'sl')
        if win != -1:
            return win
    return -1

def check4in5(board, row, col, board_size, direction):  # 如果5个棋子里有4个对面的棋子，剩下是空位
    if check_line_in_board(row, col, board_size, direction, 5) is True:
        line = get5(board, row, col, direction)
        if sum(line) == -4:
            return check_space5(board, row, col, board_size, direction)  # 下在空位的位置
    return -1

def check3in4(board, row, col, board_size, direction):
    if check_line_in_board(row, col, board_size, direction, 4) is True:
        # 获取一行里的4个棋子
        line = get4(board, row, col, direction)
        if (sum(line) == -3) and (line[0] == -1) and (line[3] == -1):
            # 检测头尾有没有被挡住
            start_block = False
            end_block = False
            if direction == 'h':
                start_block, end_block = check_block_h(board, row, col, board_size, 4, 1)
            elif direction == 'v':
                start_block, end_block = check_block_v(board, row, col, board_size, 4, 1)
            elif direction == 'sr':
                start_block, end_block = check_block_sr(board, row, col, board_size, 4, 1)
            elif direction == 'sl':
                start_block, end_block = check_block_sl(board, row, col, board_size, 4, 1)
            # 对头尾有没有被挡住做出动作
            if (start_block is False) or (end_block is False):  # 如果头尾都没有被挡住或有一头被挡住，那就下在空位的位置
                return check_space4(board, row, col, board_size, direction)  # 下在空位的位置
    return -1

def check3continuous(board, row, col, board_size, direction):
    if check_line_in_board(row, col, board_size, direction, 3) is True:
        line = get3(board, row, col, direction)

        if sum(line) == -3:  # 如果对方3个棋子连起来
            start_block, end_block = check_block_next(board, row, col, board_size, direction, 3)
            start_block2, end_block2 = check_block_one_away(board, row, col, board_size, direction, 3)

            if (start_block is False) and (end_block is False):
                if start_block2 is False:
                    return get_line_head(row, col, board_size, direction)
                elif end_block2 is False:
                    return get_line_tail(row, col, board_size, direction, 3)  # 根据尽量不下，保留进攻机会的原则，搞出的瞎搞算法
    return -1  # 如果不符合上面的情况，就不下棋

def check_line(board, row, col, board_size, direction):
    # 检查5个棋子里面有4个对面的
    c4in5 = check4in5(board, row, col, board_size, direction)
    if c4in5 != -1:
        return c4in5
    # 检查4个棋子里面有3个对面的
    c3in4 = check3in4(board, row, col, board_size, direction)
    if c3in4 != -1:
        return c3in4
    # 检查3个连续的子
    con3 = check3continuous(board, row, col, board_size, direction)
    if con3 != -1:
        return con3
    return -1

def check_line_general(board, row, col, board_size):  # 检查在单行内的变化
    # 横向
    lineh = check_line(board, row, col, board_size, 'h')
    if lineh != -1:
        return lineh
    # 纵向
    linev = check_line(board, row, col, board_size, 'v')
    if linev != -1:
        return linev
    # 斜右下
    linesr = check_line(board, row, col, board_size, 'sr')
    if linesr != -1:
        return linesr
    # 斜左下
    linesl = check_line(board, row, col, board_size, 'sl')
    if linesl != -1:
        return linesl
    return -1

def check_multiline_h(board, row, col, board_size):  # 横向
    for i in range(4):
        start_row = row
        start_col = col - i
        if check_line(board, start_row, start_col, board_size, 'h') != -1:
            return True
    return False

def check_multiline_v(board, row, col, board_size):  # 纵向
    for i in range(4):
        start_row = row - i
        start_col = col
        if check_line(board, start_row, start_col, board_size, 'v') != -1:
            return True
    return False

def check_multiline_sr(board, row, col, board_size):  # 左上到右下
    for i in range(4):
        start_row = row - i
        start_col = col - i
        if check_line(board, start_row, start_col, board_size, 'sr') != -1:
            return True
    return False

def check_multiline_sl(board, row, col, board_size):  # 右上到左下
    for i in range(4):
        # 右上到左下
        start_row = row - i
        start_col = col + i
        if check_line(board, start_row, start_col, board_size, 'sl') != -1:
            return True
    return False

def check_multiline(board, row, col, board_size):
    state = []
    state.append(check_multiline_h(board, row, col, board_size))
    state.append(check_multiline_v(board, row, col, board_size))
    state.append(check_multiline_sr(board, row, col, board_size))
    state.append(check_multiline_sl(board, row, col, board_size))
    # 检查有几个方向被判定为要下棋的
    num_true = 0
    for s in state:
        if s is True:
            num_true += 1
    # 如果要下棋的方向等于或超过2个，那就下棋
    if num_true >= 2:
        return rowcol2step(row, col, board_size)
    return -1

def check_multiline_general(board, row, col, board_size):
    if board[row][col] == 0:
        board_copy = board.copy()
        board_copy[row][col] = -1
        return check_multiline(board_copy, row, col, board_size)
    return -1

'''##################################################################################################################'''

def check3in4_attack1(board, row, col, board_size, direction):  # 要赢的节奏
    if check_line_in_board(row, col, board_size, direction, 4) is True:
        # 获取一行里的4个棋子
        line = get4(board, row, col, direction)
        if (sum(line) == 3) and (line[0] == 1) and (line[3] == 1):
            # 检测头尾有没有被挡住
            start_block = False
            end_block = False
            if direction == 'h':
                start_block, end_block = check_block_h(board, row, col, board_size, 4, 1)
            elif direction == 'v':
                start_block, end_block = check_block_v(board, row, col, board_size, 4, 1)
            elif direction == 'sr':
                start_block, end_block = check_block_sr(board, row, col, board_size, 4, 1)
            elif direction == 'sl':
                start_block, end_block = check_block_sl(board, row, col, board_size, 4, 1)
            # 对头尾有没有被挡住做出动作
            if (start_block is False) and (end_block is False):  # 如果头尾都被挡住，那就pass
                return check_space4(board, row, col, board_size, direction)  # 下在空位的位置
    return -1

def check3continuous_attack1(board, row, col, board_size, direction):  # 要赢的节奏
    if check_line_in_board(row, col, board_size, direction, 3) is True:
        line = get3(board, row, col, direction)

        if sum(line) == 3:  # 如果自己3个棋子连起来
            start_block, end_block = check_block_next(board, row, col, board_size, direction, 3)
            start_block2, end_block2 = check_block_one_away(board, row, col, board_size, direction, 3)

            if (start_block is False) and (end_block is False):
                if start_block2 is False:
                    return get_line_head(row, col, board_size, direction)
                elif end_block2 is False:
                    return get_line_tail(row, col, board_size, direction, 3)
    return -1  # 如果不符合上面的情况，就不下棋

def check_line_attack1(board, row, col, board_size, direction):
    # 检查4个棋子里面有3个自己的棋子
    c3in4 = check3in4_attack1(board, row, col, board_size, direction)
    if c3in4 != -1:
        return c3in4
    # 检查自己有没有3个连续的子
    con3 = check3continuous_attack1(board, row, col, board_size, direction)
    if con3 != -1:
        return con3
    return -1

def check_line_general_attack1(board, row, col, board_size):  # 检查在单行内的变化
    # 横向
    lineh = check_line_attack1(board, row, col, board_size, 'h')
    if lineh != -1:
        return lineh
    # 纵向
    linev = check_line_attack1(board, row, col, board_size, 'v')
    if linev != -1:
        return linev
    # 斜右下
    linesr = check_line_attack1(board, row, col, board_size, 'sr')
    if linesr != -1:
        return linesr
    # 斜左下
    linesl = check_line_attack1(board, row, col, board_size, 'sl')
    if linesl != -1:
        return linesl
    return -1

def check_multiline_general_attack1(board, row, col, board_size):
    board = board * (-1)
    return check_multiline_general(board, row, col, board_size)

def check_line_general_attack2(board, row, col, board_size):
    board = board * (-1)
    if board[row][col] == 0:
        board_copy = board.copy()
        board_copy[row][col] = -1
        if check_line_general(board_copy, row, col, board_size) != -1:
            return rowcol2step(row, col, board_size)
    return -1

# 为接下来的进攻做准备
def check2continuous_attack3(board, row, col, board_size, direction):  # 检测到有两个子连着
    if check_line_in_board(row, col, board_size, direction, 2) is True:
        line = get2(board, row, col, direction)

        if sum(line) == 2:  # 如果自己2个棋子连起来
            start_block, end_block = check_block_next(board, row, col, board_size, direction, 2)
            start_block2, end_block2 = check_block_one_away(board, row, col, board_size, direction, 2)

            if (start_block is False) and (end_block is False):
                if start_block2 is False:
                    return get_line_head(row, col, board_size, direction)
                elif end_block2 is False:
                    return get_line_tail(row, col, board_size, direction, 2)
    return -1  # 如果不符合上面的情况，就不下棋

def check2continuous_general_attack3(board, row, col, board_size):  # 检查在单行内的变化
    # 横向
    lineh = check2continuous_attack3(board, row, col, board_size, 'h')
    if lineh != -1:
        return lineh
    # 纵向
    linev = check2continuous_attack3(board, row, col, board_size, 'v')
    if linev != -1:
        return linev
    # 斜右下
    linesr = check2continuous_attack3(board, row, col, board_size, 'sr')
    if linesr != -1:
        return linesr
    # 斜左下
    linesl = check2continuous_attack3(board, row, col, board_size, 'sl')
    if linesl != -1:
        return linesl
    return -1

def check_opposite_density(board, row, col, board_size, range_size):  # 检查以一个子为中心一定范围内对方棋子的密度怎么样
    start_row = row - int(range_size / 2)
    start_col = col - int(range_size / 2)
    opposite = 0
    total = range_size * range_size
    for i in range(range_size):
        for j in range(range_size):
            row = start_row + i
            col = start_col + j
            if check_in_board(row, col, board_size) is True:
                if board[row][col] == -1:  # 如果棋子
                    opposite = opposite + 1
            else:
                opposite = opposite + 1
    return opposite / total

def check_opposite_density_in_5x5(board, row, col, board_size):
    return check_opposite_density(board, row, col, board_size, 5)

def take_density(element):  # 只是为了排序，没别的用
    return element[2]

def take_distance(element):  # 只是为了排序，没别的用
    return element[2]

def get_8_around(board, row, col, board_size):  # 获取一个位置周围能下的位置
    position_list = [[row - 1, col - 1],
                     [row - 1, col],
                     [row - 1, col + 1],
                     [row, col - 1],
                     [row, col + 1],
                     [row + 1, col - 1],
                     [row + 1, col],
                     [row + 1, col + 1]]
    point_list = []
    for position in position_list:
        row = position[0]
        col = position[1]
        if check_in_board(row, col, board_size) and board[row][col] == 0:
            point_list.append([row, col])
    return point_list

def get_distance_to_center(row, col, board_size):
    center = (board_size - 1) // 2
    distance = math.sqrt((row - center)**2 + (col - center)**2)
    return distance

def check1_attack3(board, board_size):  # 棋盘上都是单独的子
    density_list = []
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == 1:
                density = check_opposite_density_in_5x5(board, row, col, board_size)
                density_list.append([row, col, density])
    density_list.sort(key=take_density, reverse=False)  # 选出周围对面的棋子最少的一个子

    for element in density_list:
        row = element[0]
        col = element[1]
        point_list = get_8_around(board, row, col, board_size)
        distance_list = []
        for point in point_list:
            p_row = point[0]
            p_col = point[1]
            distance = get_distance_to_center(p_row, p_col, board_size)
            distance_list.append([p_row, p_col, distance])
        distance_list.sort(key=take_distance, reverse=True)  # 要下的棋子的位置离中心越远越好
        if distance_list != []:
            p_row = distance_list[0][0]
            p_col = distance_list[0][1]
            return rowcol2step(p_row, p_col, board_size)
    return -1

def check_no_piece(board, board_size):
    no_piece = True
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == 1:  # 如果棋盘上有一个自己的棋子，那就跳过
                no_piece = False
    if no_piece is True:  # 从中间一块随机选一个点出来
        half_board_size = board_size // 2
        while True:
            row = random.randint(int(half_board_size - (half_board_size // 2)), int(half_board_size + (half_board_size // 2)))
            col = random.randint(int(half_board_size - (half_board_size // 2)), int(half_board_size + (half_board_size // 2)))
            if board[row][col] == 0:
                return rowcol2step(row, col, board_size)
    else:
        return -1

def check4in5_all_direction(board, row, col, board_size):
    # 横向
    c4in5h = check4in5(board, row, col, board_size, 'h')
    if c4in5h != -1:
        return c4in5h
    # 纵向
    c4in5v = check4in5(board, row, col, board_size, 'v')
    if c4in5v != -1:
        return c4in5v
    # 斜右下
    c4in5sr = check4in5(board, row, col, board_size, 'sr')
    if c4in5sr != -1:
        return c4in5sr
    # 斜左下
    c4in5sl = check4in5(board, row, col, board_size, 'sl')
    if c4in5sl != -1:
        return c4in5sl
    return -1

def check3in4_all_direction(board, row, col, board_size):
    # 横向
    c3in4h = check3in4(board, row, col, board_size, 'h')
    if c3in4h != -1:
        return c3in4h
    # 纵向
    c3in4v = check3in4(board, row, col, board_size, 'v')
    if c3in4v != -1:
        return c3in4v
    # 斜右下
    c3in4sr = check3in4(board, row, col, board_size, 'sr')
    if c3in4sr != -1:
        return c3in4sr
    # 斜左下
    c3in4sl = check3in4(board, row, col, board_size, 'sl')
    if c3in4sl != -1:
        return c3in4sl
    return -1

def check3continuous_all_direction(board, row, col, board_size):
    con3h = check3continuous(board, row, col, board_size, 'h')
    if con3h != -1:
        return con3h
    # 纵向
    con3v = check3continuous(board, row, col, board_size, 'v')
    if con3v != -1:
        return con3v
    # 斜右下
    con3sr = check3continuous(board, row, col, board_size, 'sr')
    if con3sr != -1:
        return con3sr
    # 斜左下
    con3sl = check3continuous(board, row, col, board_size, 'sl')
    if con3sl != -1:
        return con3sl
    return -1

def get_action(env, side, board_size):  # 下棋
    board = env.board
    if side == 0:
        board = board * (-1)  # 如果现在是白棋在下，那么就把盘面转成黑棋的

    # 棋盘上没有棋
    no_piece = check_no_piece(board, board_size)
    if no_piece != -1:
        return no_piece

    for row in range(board_size):
        for col in range(board_size):
            # 检查是不是直接赢了
            win = check_win_all_direction(board, row, col, board_size)
            if win != -1:
                return win

    for row in range(board_size):
        for col in range(board_size):
            # 防守4个子
            c4in5 = check4in5_all_direction(board, row, col, board_size)
            if c4in5 != -1:
                return c4in5

    for row in range(board_size):
        for col in range(board_size):
            # 进攻3变4
            line_general_attack1 = check_line_general_attack1(board, row, col, board_size)
            if line_general_attack1 != -1:
                return line_general_attack1

    for row in range(board_size):
        for col in range(board_size):
            # 防守3变4
            c3in4 = check3in4_all_direction(board, row, col, board_size)
            if c3in4 != -1:
                return c3in4
            con3 = check3continuous_all_direction(board, row, col, board_size)
            if con3 != -1:
                return con3

    for row in range(board_size):
        for col in range(board_size):
            # 进攻检查多行的变化
            multiline_general_attack1 = check_multiline_general_attack1(board, row, col, board_size)
            if multiline_general_attack1 != -1:
                return multiline_general_attack1

    for row in range(board_size):
        for col in range(board_size):
            # 防守检查多行的变化
            multiline_general = check_multiline_general(board, row, col, board_size)
            if multiline_general != -1:
                return multiline_general

    # 试探性攻击
    for row in range(board_size):
        for col in range(board_size):
            # 检查单行的变化
            line_general_attack2 = check_line_general_attack2(board, row, col, board_size)
            if line_general_attack2 != -1:
                return line_general_attack2

    # 为接下来的进攻做准备
    for row in range(board_size):
        for col in range(board_size):
            # 检查单行的变化
            con2 = check2continuous_general_attack3(board, row, col, board_size)
            if con2 != -1:
                return con2
    
    # 全是单独的子
    one = check1_attack3(board, board_size)
    if one != -1:
        return one

    return env.random_action()
