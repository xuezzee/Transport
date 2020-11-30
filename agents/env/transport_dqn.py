#coding=utf-8
import random
from env.gridgame import GridGame
# from myagent import my_agent`
from itertools import count
from gym.spaces import Discrete, Box
import numpy as np
import re
import tkinter
from PIL import ImageDraw, ImageFont
class Material():
    def __init__(self,position,number=0):  # number means 货量
        self.position=position
        self.number=number
    def update(self):
        x=random.randint(-1,1)
        self.number=self.number+x
#     # def look(self):
# #         # print ("material look:")
#         # print (self.position,self.number)
class Car():
    def __init__(self,position,number=0):
        self.position=position
        self.number=number
        self.max_num=2
    #def load(self):#

    #def unload(self):#

#     # def look(self):
#         # print ("car look:")
        # print (self.position,self.number)
class Plane():
    def __init__(self,position,number=0):
        self.position=position
        self.number=number
        self.max_num=1
    #def load(self):#

    #def unload(self):#

#     # def look(self):
#         # print ("plane look:")
        # print (self.position,self.number)
class Barriers():
    def __init__(self,position):
        self.position=position
        self.total=random.randint(1,4)#total time
        self.age=0#time now
        # print ("set a new barrier:")
    #     self.look()
    # # def look(self):
        # print (self.position,self.total,self.age)

class Opponent():
    def __init__(self):
        self.barriers=0#the setting barriers ,initial=0
        self.barriers_list=[]
    def add_barriers(self,pos):
        self.barriers+=1
        self.barriers_list.append(Barriers(position=pos))
    def update_barriers(self):
        # print("update_barriers fuction:")
        free_list=[]
        l=len(self.barriers_list)
        for i in range(l-1,-1,-1):#[self.barriers-1,......,0]
            now_bar=self.barriers_list[i]
            now_bar.age+=1
            # print ("the %dth:"%i)
            # now_bar.look()
            if(now_bar.age==now_bar.total):
                free_list.append(now_bar.position)
                self.barriers_list.pop(i)
                self.barriers-=1
        # print ("before number:%d later number:%d"%(l,self.barriers))
        # print ("free list:",free_list)
        return free_list
class Transport(GridGame):
    def __init__(self,conf):#(self,n_player=2, board_width=15, board_height=15, n_cell_type=5, materials=4, cars=5, planes=5,barriers=10,max_step=1000,game_name="Materials Transport",K=1):
        self.conf = conf
        colors = conf.get('colors',[(255,255,255),(255,255,0),(138,43,226),(0,255,255),(0,255,0),(255,69,0)])
        #n_cell_type :0 NULL,1 fixed barrier,2 opponet set barrier 3. materials points 4:cars

        self.board_height=int(conf['board_height'])
        self.board_width=int(conf['board_width'])
        self.materials=int(conf['materials'])#the number of materials
        self.cars=int(conf['cars'])
        self.planes=int(conf['planes'])
        self.barriers=int(conf['barriers'])#the number of fix barriers
        self.max_step=int(conf['max_step'])#the max game time
        self.step_cnt=0
        self.map_path=conf['map_path']
        self.initial_map=self.read_data()
        super(Transport, self).__init__(conf,colors)

        self.actions = [0,1,2,3,4]
        self.actionx = [-1, 0, 1, 0,0]
        self.actiony = [0, 1, 0, -1,0]
        self.actions_name = {0: "up", 1: "right", 2: "down", 3: "left",4:"stop"}
        self.K=int(conf['K']) #for update materials

        self.map=[[0] * self.board_width for _ in range(self.board_height)]
        self.fix_map=[[0] * self.board_width for _ in range(self.board_height)]#fixed barriers  materials points
        self.car_map=[[0] * self.board_width for _ in range(self.board_height)]#to record how many cars in every position

        self.free_position=[]
        self.fixed_barrier=[]
        self.material_points=[]
        self.map_mate_index={}

        self.set_fixed_barriers()
        self.set_materials()
        self.cars_list=[]
        self.set_cars()


        self.planes_list=[]
        self.set_planes()
        self.opponent=Opponent()
        self.reward=0
        self.reward_per = np.zeros(self.cars) #TODO: TO CHECK
        self.root = None

    # TODO: UPDATE
    @property  # 直接掉一个变量 eg. env.observation_space
    def observation_space(self):
        return [Box(low=0, high=50, shape=(121,1)) for i in range(self.cars)]
    @property  # 直接掉一个变量 eg. env.observation_space
    def action_space(self):
        return [Discrete(5) for i in range(self.cars)]
    @property
    def agents(self):
        return ['agent' for i in range(self.cars)]

    def read_data(self):
        data=[]
        with open(self.map_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                data.append([int(line[0]),int(line[1])])
                # print(line)
        return data

    def is_terminal(self):
        # print(self.reward_tot)
        return self.step_cnt > self.max_step or self.reward_tot == 2
    #
    # def is_a_g_terminal(self):
    #     for i in range(self.materials):
    #         num = self.material_points[i].number
    #         if num == 0:
    #             continue
    #         else:
    #             return False
    #     return True

    def set_fixed_barriers(self):
        #input
        for i in range(self.barriers):
#             #print ("请输入第%d个固定障碍的坐标：'x,y'"%i)
            #a=input()
            #x,y=a.split(' ')
            #x=int(x)
            #y=int(y)
            pos=self.initial_map[i]
            x=pos[0]
            y=pos[1]
            self.fixed_barrier.append([x,y])
            self.map[x][y]=1
        # self.look()
        #random#
        '''
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.free_position.append([i,j])
        random.shuffle(self.free_position)
        for i in range(self.barriers):
            pos=self.free_position[0]
            self.fixed_barrier.append(pos)
            x=pos[0]
            y=pos[1]
            self.map[x][y]=1
            self.free_position.pop(0)#delete index
        '''
#         #    print(x,y) #print to test
        #for i in range(self.board_height):
#         #    print(self.map[i])#for j in range(self.board_width):
    def set_materials(self):
        # while True:
        #     num1 = random.randint(-4, 4)
        #     num2 = random.randint(-4, 4)
        #     num3 = random.randint(-4, 4)
        #     num4 = random.randint(-4, 4)
        #     if num1 + num2 + num3 + num4 == 0:
        #         temp = [num1, num2, num3, num4]
        #         break

        temp = [ 8, -8, 0, 0]

        for i in range(self.materials):
#             #print ("请输入第%d个物资集散点的坐标：'x,y'"%i)
            #a=input()
            #x,y=a.split(' ')
            pos=self.initial_map[i+self.barriers]
            x=pos[0]
            y=pos[1]
            point = Material(position=[x,y], number=temp[i])
            # point = Material(position=[x,y], number=0)
            self.material_points.append(point)
            self.map_mate_index[(x,y)]=i
            self.map[x][y]=3
        for i in range(self.board_height):
            for j in range(self.board_width):
                if(self.map[i][j]==0):
                    self.free_position.append([i, j])
        # self.look()
        '''
        #random
        random.shuffle(self.free_position)
        for i in range(self.materials):
            pos = self.free_position[0]
            point=Material(position=pos,number=0)
            self.material_points.append(point)
            x = pos[0]
            y = pos[1]
            self.map[x][y] = 3
            self.free_position.pop(0)
            self.map_mate_index[(x,y)]=i
        self.fix_map=self.map
#         #print for test
        '''
        '''
            # print (x,y)
            # self.material_points[i].look()
        for i in range(self.board_height):
            # print(self.map[i])#for j in range(self.board_width):
        '''
    def set_cars(self):
        # random.shuffle(self.free_position)

        for i in range(self.cars):
            pos = self.free_position[0]

            point = Car(position=pos, number=0)
            self.cars_list.append(point)
            x = pos[0]
            y = pos[1]
            self.map[x][y] = 4#cars
            self.car_map[x][y]+=1
            self.free_position.pop(0)
#         # print for test
        '''
            # print (x,y)
            # self.cars_list[i].look()
        for i in range(self.board_height):
            # print(self.map[i])#for j in range(self.board_width):
        '''
    def set_planes(self):
        x=[]
        for i in range(self.board_height):
            for j in range(self.board_width):
                x.append([i,j])
        random.shuffle(x)
        for i in range(self.planes):
            pos = x[i]
            point = Plane(position=pos, number=0)
            self.planes_list.append(point)
#          #   self.planes_list[i].look()

    # add reset
    def reset(self):
        # self.__init__(self.conf)

        self.step_cnt = 0

        self.map = [[0] * self.board_width for _ in range(self.board_height)]
        self.fix_map = [[0] * self.board_width for _ in range(self.board_height)]  # fixed barriers  materials points
        self.car_map = [[0] * self.board_width for _ in
                        range(self.board_height)]  # to record how many cars in every position

        self.free_position = []
        self.fixed_barrier = []
        self.material_points = []
        self.map_mate_index = {}

        self.set_fixed_barriers()
        self.set_materials()
        self.cars_list = []
        self.set_cars()

        self.planes_list = []
        self.set_planes()
        self.opponent = Opponent()
        self.reward = 0
        self.reward_per = np.zeros(self.cars)  # TODO: TO CHECK
        self.reward_tot = 0


        #####################################
        # add: state modified
        # state = []
        # for i in range(self.board_height):
        #     for j in range(self.board_width):
        #         state.append(self.map[i][j])
        #
        # state_int = [ np.array(state) for i in range(self.cars)]
        #


        # 之前的
        state_int = [] #np.zeros((self.cars, self.board_width * self.board_height))  # TODO: TO CHECK
        state = []
        
        ######################################
        for i in range(self.board_height):
            for j in range(self.board_width):
                state_int.append(self.map[i][j])
        ######################################

        vehicle_num = len(self.cars_list)
        for n in range(vehicle_num):  # 3 * 2 = 6
            # pos = self.cars_list[n].position
            # state_int.append(pos[0])
            # state_int.append(pos[1])
            state_int.append(self.cars_list[n].number)
        # for m in range(self.materials):  # 3 * 4 = 12
        #     # pos = self.material_points[m].position
        #     # state_int.append(pos[0])
        #     # state_int.append(pos[1])
        #     state_int.append(self.material_points[m].number)

        return state_int


        # return self.get_observation_s(self.current_state)
        # return self.current_state #,info_after #add

    def get_next_state(self,joint_action):

        free_barriers_list=self.opponent.update_barriers()#update opponent barriers
        for i in range(len(free_barriers_list)):
            pos=free_barriers_list[i]
            self.map[pos[0]][pos[1]]=0
#         # game.look()
        # print ("setting new barriers function")

        x=joint_action[0][0].index(1)
        y=joint_action[1][0].index(1)

        # print (x,y,self.board_height,self.board_width)
        if ((x < 0) or (x >= self.board_height) or(y < 0) or (y >=self.board_width)):
            print("敌方放置障碍位置在地图外")
        else:

            # if (self.map[x][y] != 0):
            #     print("敌方放置障碍位置非空")
            # else:

            if (self.map[x][y] == 0):
                self.opponent.add_barriers([x, y])
                self.map[x][y] = 2
#         # game.look()

        # print ("cars action")
        #car_action=joint_action[1]
        for i in range(self.cars):
            # print ("car%d start:"%i)

            car_now=self.cars_list[i]
            # # print ("car_now.look")
            # car_now.look()
            # # game.look()
            old_pos=car_now.position
            old_x=old_pos[0]
            old_y=old_pos[1]
            check,new_pos =self.check_car_action(car_now.position,joint_action[2+i][0].index(1))
            x=new_pos[0]
            y=new_pos[1]
            if(check):
                car_now.position=new_pos
                self.car_map[x][y] += 1
                self.car_map[old_x][old_y] -= 1
                if(self.map[x][y]==0):
                    self.map[x][y]=4

                if(self.car_map[old_x][old_y]==0 and self.map[old_x][old_y]==4):#所有汽车都移走了
                    self.map[old_x][old_y]=0
            # print ()
#             # game.look()
#             # print ("car_now.look")

#             # car_now.look()
            # print ("car%d end:" % i)

        # print ("planes action")
        #plane_action = joint_action[2]
        # print("plane action:")
        for i in range(self.planes):
            # print ("plane%d start:" % i)

            plane_now = self.planes_list[i]
            # # print ("plane_now.look")
            # plane_now.look()
            # # game.look()

            check, new_pos = self.check_plane_action(plane_now.position, joint_action[2+self.cars+i][0].index(1))

            if (check):
                plane_now.position = new_pos


#             # print ("plane_now.look")

            # plane_now.look()
            # print ("plane%d end:" % i)

        # print("update materials points")
        '''
        if((self.step_cnt+1)%self.K==0):
            for i in range(self.materials):
                # print ("material%d:"%i)
                m_now=self.material_points[i]
                # m_now.look()
                m_now.update()
        '''
                # m_now.look()


        # print("load and unload cars and planes")
        vehicle_list=self.cars_list+self.planes_list
        for i in range(self.cars+self.planes):
            vehicle=vehicle_list[i]
            pos=vehicle.position
            x=pos[0]
            y=pos[1]
            # vehicle.look()
            if(self.map[x][y]==3):#material points
                index=self.map_mate_index[(x,y)]
                material_now=self.material_points[index]
                # material_now.look()
                material_num=material_now.number
                if(material_num>0):#load
                    exchange=min(material_num,vehicle.max_num-vehicle.number)
                    vehicle.number=vehicle.number+exchange
                    material_num=material_num-exchange
                    material_now.number=material_num
                    # self.reward += exchange
                else:
                    if(material_num<0):#unload
                        exchange = min(-material_num, vehicle.number)
                        vehicle.number = vehicle.number - exchange
                        material_num = material_num + exchange
                        self.reward += exchange
                        self.reward_per += exchange
                        material_now.number=material_num
            #     material_now.look()
            # vehicle.look()

        self.step_cnt+=1
        # if self.is_a_g_terminal():
        #     self.reset()
        # else:
        for i in range(self.board_height):
            for j in range(self.board_width):
                # print(self.current_state)
                self.current_state[i][j][0]=self.map[i][j]
        for i in range(self.planes):
            pos=self.planes_list[i].position
            x=pos[0]
            y=pos[1]
            if(self.current_state[x][y][0]==0):
                self.current_state[x][y][0] =5
        info_after=''#to add later

        return self.current_state, self.get_observation_s(self.current_state)

    def get_observation_s(self, s): # from old current state to state update
        # observation shape: 121
        s_ = s
        state_id = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                state_id.append(s_[i][j][0])

        vehicle_num = len(self.cars_list)
        for n in range(vehicle_num):  # 3 * 2 = 6
            # pos = self.cars_list[n].position
            # state.append(pos[0])
            # state.append(pos[1])
            state_id.append(self.cars_list[n].number)

        return state_id

    def get_observation_small(self, s):
        s_ = s
        state_id = []
        vehicle_num = len(self.cars_list)
        for n in range(vehicle_num):  # 3 * 2 = 6
            pos = self.cars_list[n].position
            state_id.append(pos[0])
            state_id.append(pos[1])
            state_id.append(self.cars_list[n].number)
        return state_id

    def get_observation_space(self): #TODO
        return 121


    def get_observation_add(self, s, agent_id): # from old current state to state update

        '''
        utility: mapping from state to observation
        return obs1, obs2

        TODO: different obs
        '''

        s_ = s
        state = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                state.append(s_[i][j][0])

        vehicle_num = len(self.cars_list)
        for n in range(vehicle_num): # 3 * 2 = 6
            # pos = self.cars_list[n].position
            # state.append(pos[0])
            # state.append(pos[1])
            state.append(self.cars_list[n].number)
        for m in range(self.materials): # 3 * 4 = 12
            # pos = self.material_points[m].position
            # state.append(pos[0])
            # state.append(pos[1])
            state.append(self.material_points[m].number)

        return state

    def check_car_action(self,position,action):
        x=position[0]
        y=position[1]
        x=x+self.actionx[action]
        y=y+self.actiony[action]
        pos=[x,y]
        check=1
        # penalty = 0
        if ((x < 0) or (x >= self.board_height) or(y < 0) or (y >=self.board_width)):
            # print("汽车行驶超过地图外")
            check=0
            penalty = 1
            return check,position #, penalty
        if(self.map[x][y]==1 or self.map[x][y]==2):
            # print("汽车行驶遇到障碍物")
            check = 0
            penalty = 1
            return check, position#, penalty
        return check,pos #, penalty

    def check_plane_action(self,position,action):
        x=position[0]
        y=position[1]
        x=x+self.actionx[action]
        y=y+self.actiony[action]
        pos=[x,y]
        check=1
        if ((x < 0) or (x >= self.board_height) or(y < 0) or (y >=self.board_width)):
            # print("飞机行驶超过地图外")
            check=0
            return check,position
        return check,pos
    def get_terminal_actions(self):
        # print("请输入个敌方放置障碍的坐标'x y'，加引号并且空格隔开：" )
        a=input()
        actions=[]
        x, y = a.split(' ')
        actions.append([int(x),int(y)])
        # print("请输入%d辆汽车的动作，0上，1 右，2下，3左,4不动'a b c d'"%self.cars)
        a=input()
        car_action=a.split(' ')
        for i in range(len(car_action)):
            car_action[i]=int(car_action[i])
        actions.append(car_action)
        # print("请输入%d架飞机的动作，0上，1 右，2下，3左,4不动'a b c d'" % self.planes)
        a = input()
        plane_action = a.split(' ')
        for i in range(len(plane_action)):
            plane_action[i] = int(plane_action[i])
        actions.append(plane_action)
        return actions
    def is_terminal(self):
        # print(self.step_cnt, self.reward_tot)
        # print('reward_tot', self.reward_tot)
        return self.step_cnt > self.max_step or self.reward_tot == 8
    # def look(self):
    #     # print ("game look")

    #     for i in range(self.board_height):
    #         # print(self.map[i])

    def set_action_space(self):#add
        action_space = [[self.board_height],[self.board_width]]+[[5] for _ in range(self.cars)]+[[5] for _ in range(self.planes)]
        #对手设置的坐标，汽车的动作，飞机的动作
        return action_space
    def step_before_info(self, info=''):#add
        setting_barriers_pos=[]
        setting_barriers_total=[]
        setting_barriers_age=[]

        for i in range (len(self.opponent.barriers_list)):
            setting_barriers_pos .append(self.opponent.barriers_list[i].position)
            setting_barriers_total.append(self.opponent.barriers_list[i].total)
            setting_barriers_age.append(self.opponent.barriers_list[i].age)
        info = "当前敌方设置障碍物位置:%s" % str(setting_barriers_pos)

        info += "\n当前敌方设置障碍物年限：%s" % str(setting_barriers_total)
        info += "\n当前敌方设置障碍物年龄：%s" % str(setting_barriers_age)

        material_pos=[]
        material_num=[]
        for i in range(self.materials):
            material_pos.append(self.material_points[i].position)
            material_num.append(self.material_points[i].number)

        info += "\n当前物资集散点位置：%s" % str(material_pos)
        info += "\n当前物资集散点物资数量：%s" % str(material_num)


        return info
    def get_reward(self, joint_action):#add
        # reward = [self.reward_per]
        reward = [np.array([self.reward, self.reward])]
        self.reward = 0
        self.reward_per = np.zeros(self.cars)
#         # print("score:", self.won)
        # return [self.reward,-self.reward]  #TODO: reward 需不需要reset to 0
        self.reward_tot += self.reward
        return reward

    def check_win(self):#
        if(self.reward>0):
            # print("我方胜")
            return 1
        if(self.reward<0):
            # print("敌方胜")
            return -1
        # print("平局")
        return 0
    @staticmethod
    def _render_board(state, board, colors, unit, fix, extra_info):##add
        # print("1111111111111111111111111")
        im = GridGame._render_board(state, board, colors, unit, fix)
        draw = ImageDraw.Draw(im)
        # fnt = ImageFont.truetype("Courier.dfont", 16)
        for i in extra_info.keys():
            x=i[0]
            y=i[1]
            draw.text(((y + 1.0 / 4) * unit, (x + 1.0 / 4) * unit),
                      extra_info[i],
                      # font=fnt,
                      fill=(0, 0, 0))

        '''
        for i, pos in zip(count(1), extra_info):
            x, y = pos
            draw.text(((y + 1.0 / 4) * unit, (x + 1.0 / 4) * unit),
                      "#{}".format(i),
                      font=fnt,
                      fill=(0, 0, 0))
        '''
        return im

    def render_board(self):##add
        extra_info = {}
        for i in range(self.barriers):
            pos=self.fixed_barrier[i]
            x=pos[0]
            y=pos[1]
            if (x,y) not in extra_info.keys():
                extra_info[(x,y)]='F_B'
            else:
                extra_info[(x, y)] += '\n'+'F_B'
        for i in range(self.opponent.barriers):
            pos=self.opponent.barriers_list[i].position
            x=pos[0]
            y=pos[1]
            if (x,y) not in extra_info.keys():
                extra_info[(x,y)]='Set_B'
            else:
                extra_info[(x, y)] += '\n'+'Set_B'
        for i in range(self.materials):
            pos=self.material_points[i].position
            num=self.material_points[i].number
            x=pos[0]
            y=pos[1]
            if (x,y) not in extra_info.keys():
                extra_info[(x,y)]='M'+str(num)
            else:
                extra_info[(x, y)] += '\n'+'M'+str(num)
        for i in range(self.cars):
            pos = self.cars_list[i].position
            num = self.cars_list[i].number
            x = pos[0]
            y = pos[1]
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'C' +str(i)+':'+ str(num)
            else:
                extra_info[(x, y)] += '\n'+'C' +str(i)+':'+ str(num)
        for i in range(self.planes):
            pos = self.planes_list[i].position
            num = self.planes_list[i].number
            x = pos[0]
            y = pos[1]
            if (x, y) not in extra_info.keys():
                extra_info[(x, y)] = 'P' +str(i)+':'+ str(num)
            else:
                extra_info[(x, y)] += '\n'+'P' +str(i)+':'+ str(num)

        im_data = np.array(
            Transport._render_board(self.get_render_data(self.current_state), self.grid, self.colors, self.grid_unit, self.grid_unit_fix,
                                        extra_info))
        self.game_tape.append(im_data)
        return im_data

    def _close_view(self):
        if self.root:
            self.root.destory()
            self.root = None
            self.canvas = None
        # self.done = True

    def _render(self):
        map = np.array(self.map)
        scale = 30
        width = map.shape[0] * scale
        height = map.shape[1] * scale
        if self.root is None:
            self.root = tkinter.Tk()
            self.root.title("escalator env")
            self.root.protocol("WM_DELETE_WINDOW", self._close_view)
            self.canvas = tkinter.Canvas(self.root, width=width, height=height)
            self.canvas.pack()

        self.canvas.delete(tkinter.ALL)
        self.canvas.create_rectangle(0, 0, width, height, fill="black")

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * scale,
                y * scale,
                (x + 1) * scale,
                (y + 1) * scale,
                fill=color
            )

        for x in range(map.shape[0]):
            for y in range(map.shape[1]):
                if map[x,y] == 1:
                    fill_cell(x,y,'Grey')
                if map[x,y] == 2:
                    fill_cell(x,y,'Green')
                if map[x,y] == 3:
                    fill_cell(x,y,'Red')
                if map[x,y] == 4:
                    if self.cars_list[0].number == 0:
                        fill_cell(x,y,'Blue')
                    else:
                        fill_cell(x, y, 'Pink')

        self.root.update()

def render_game(g, fps=1):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode(g.grid.size)
    pygame.display.set_caption(g.game_name)

    clock = pygame.time.Clock()

    while not g.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # print("step %d" % g.step_cnt)
        joint_act = my_agent(g.current_state, g.joint_action_space)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        # print(joint_act)
        # print(reward)
        # print(info_before)
        pygame.surfarray.blit_array(screen, g.render_board().transpose(1, 0, 2))
        pygame.display.flip()
        if info_after:
            print(info_after)
        # 调整帧率
        clock.tick(fps)
        fname="./image/"+str(g.step_cnt)+".png"#save image
        pygame.image.save(screen, fname)

    # print("winner", g.check_win())
    # print("winner_information", str(g.won))


def create_video(step):
    import cv2

    img_root = './image/'
    fps = 1
    image = cv2.imread('../../image/1.png')
    a = image.shape
    size = (a[0], a[1])  # 1200
    #
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('./image/test.mp4', fourcc, fps, size)  #

    # for(i=1;i<471;++i)
    for i in range(1, step+1):
        frame = cv2.imread(img_root + str(i) + '.png')
        videoWriter.write(frame)
    videoWriter.release()


'''
if __name__ == "__main__":
    # game = GoBang(board_height=15, board_width=15)
    # game = Reversi(board_width=4, board_height=4)
    name = 'Materials Transport'
    conf = {
        'n_player': 2,#玩家数量
        'board_width': 11,#地图宽
        'board_height': 11,#地图高
        'n_cell_type': 5,#格子的种类
        'materials': 4,#集散点数量
        'cars': 2,#汽车数量
        'planes': 0,#飞机数量
        'barriers': 12,#固定障碍物数量
        'max_step' :30,#最大步数
        'game_name':name,#游戏名字
        'K': 5,#每个K局更新集散点物资数目
        'map_path':'map.txt',#存放初始地图
        'cell_range': 6,  # 单格中各维度取值范围（tuple类型，只有一个int自动转为tuple）##?
        'ob_board_width': None,  # 不同智能体观察到的网格宽度（tuple类型），None表示与实际网格相同##?
        'ob_board_height': None,  # 不同智能体观察到的网格高度（tuple类型），None表示与实际网格相同##?
        'ob_cell_range': None,  # 不同智能体观察到的单格中各维度取值范围（二维tuple类型），None表示与实际网格相同##?
    }


    game = Transport(conf) #Transport(n_player=2, board_width=4, board_height=4, n_cell_type=5,barriers=3,materials=4,cars=2,planes=1,max_step=100)
    game.reset()
    # render_game(game)
    # create_video(int(conf['max_step'])+1)
'''