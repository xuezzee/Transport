# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/7/10 10:24 上午   
# 描述：

from PIL import Image, ImageDraw
from itertools import count
import numpy as np
from env.game import Game

UNIT = 40
FIX = 8


class GridGame(Game):
    def __init__(self, conf, colors=None, unit_size=UNIT, fix=FIX):
        super().__init__(int(conf['n_player']))
        self.game_name = conf['game_name']
        self.max_step = int(conf['max_step'])
        self.board_width = int(conf['board_width'])
        self.board_height = int(conf['board_height'])
        self.cell_range = conf['cell_range'] if isinstance(conf['cell_range'], tuple) else (int(conf['cell_range']),)
        self.cell_dim = len(self.cell_range)
        self.cell_size = np.prod(self.cell_range)
        self.ob_board_width = conf['ob_board_width'] if not conf.get('ob_board_width') is None else [self.board_width for _ in range(self.n_player)]
        self.ob_board_height = conf['ob_board_height'] if not conf.get('ob_board_height') is None else [self.board_height for _ in range(self.n_player)]
        self.ob_cell_range = conf['ob_cell_range'] if not conf.get('ob_cell_range') is None else [self.cell_range for _ in range(self.n_player)]
        self.current_state = [[[0] * self.cell_dim for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.joint_action_space = self.set_action_space()
        self.single_action_space = self.joint_action_space[0]
        self.n_return = [0] * self.n_player

        self.won = ''

        self.grid_unit = unit_size
        self.grid = GridGame.init_board(self.board_width, self.board_height, unit_size)
        self.grid_unit_fix = fix
        self.game_tape = []
        self.colors = colors + generate_color(self.cell_size - len(colors) + 1) if not colors is None else generate_color(
            self.cell_size)
    
    def get_config(self, player_id):
        return self.ob_board_width[player_id], self.ob_board_height[player_id], self.ob_cell_range[player_id]

    def set_action_space(self):
        raise NotImplementedError

    def check_win(self):
        raise NotImplementedError
    
    def get_render_data(self, current_state):
        grid_map = [[0] * self.board_width for _ in range(self.board_height)]
        for i in range(self.board_height):
            for j in range(self.board_width):
                grid_map[i][j] = 0
                for k in range(self.cell_dim):
                    grid_map[i][j] = grid_map[i][j] * self.cell_range[k] + current_state[i][j][k]
        return grid_map

    def set_current_state(self, current_state):
        if not current_state:
            raise NotImplementedError
        not_valid = self.is_not_valid_state(current_state)
        if not not_valid:
            self.current_state = current_state

    def is_not_valid_action(self, joint_action):
        raise NotImplementedError

    def is_not_valid_state(self, current_state):
        not_valid = 0
        if len(current_state) != self.board_height or len(current_state[0]) != self.board_width or len(current_state[0][0]) != self.cell_dim:
            raise Exception("state 维度不正确！", current_state)

        for i in range(self.board_height):
            for j in range(self.board_width):
                for k in range(self.cell_dim):
                    if current_state[i][j][k] not in range(self.cell_range[k]):
                        raise Exception("state 单元值不正确！", current_state[i][j][k])

        return not_valid

    def step(self, joint_action):
        # print(joint_action)
        info_before = self.step_before_info()
        # next_state, info_after = self.get_next_state(joint_action)
        # print('ja', self.get_next_state(joint_action))

        s_original, next_state = self.get_next_state(joint_action) # added
        self.current_state = s_original
        done = self.is_terminal()
        reward = self.get_reward(joint_action)
        return next_state, reward, done, info_before

    def step_before_info(self, info=''):
        return info

    def init_action_space(self):
        joint_action = []
        for i in range(len(self.joint_action_space)):
            player = []
            for j in range(len(self.joint_action_space[i])):
                each = [0] * self.joint_action_space[i][j]
                player.append(each)
            joint_action.append(player)
        return joint_action

    def draw_board(self):
        cols = [chr(i) for i in range(65, 65 + self.board_width)]
        s = ', '.join(cols)
        print('  ', s)
        for i in range(self.board_height):
            print(chr(i + 65), self.current_state[i])

    def render_board(self):
        im_data = np.array(
            GridGame._render_board(self.get_render_data(self.current_state), self.grid, self.colors, self.grid_unit, self.grid_unit_fix))
        self.game_tape.append(im_data)
        return im_data

    @staticmethod
    def init_board(width, height, grid_unit, color=(250, 235, 215)):
        im = Image.new(mode="RGB", size=(width * grid_unit, height * grid_unit), color=color)
        draw = ImageDraw.Draw(im)
        for x in range(0, width):
            draw.line(((x * grid_unit, 0), (x * grid_unit, height * grid_unit)), fill=(105, 105, 105))
        for y in range(0, height):
            draw.line(((0, y * grid_unit), (width * grid_unit, y * grid_unit)), fill=(105, 105, 105))
        return im

    @staticmethod
    def _render_board(state, board, colors, unit, fix, extra_info=None):
        '''
            完成基本渲染棋盘操作
            设置extra_info参数仅为了保持子类方法签名的一致
        '''
        im = board.copy()
        draw = ImageDraw.Draw(im)
        for x, row in zip(count(0), state):
            for y, state in zip(count(0), row):
                if state == 0:
                    continue
                draw.rectangle(build_rectangle(y, x, unit, fix), fill=tuple(colors[state]), outline=(192, 192, 192))
        return im

    @staticmethod
    def parse_extra_info(data):
        return None


def build_rectangle(x, y, unit_size=UNIT, fix=FIX):
    return x * unit_size + unit_size // fix, y * unit_size + unit_size // fix, (x + 1) * unit_size - unit_size // fix, (
                y + 1) * unit_size - unit_size // fix


def generate_color(n):
    return [tuple(map(lambda n: int(n), np.random.choice(range(256), size=3))) for _ in range(n)]
