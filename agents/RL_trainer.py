import cv2
import time
import argparse
import numpy as np
import argparse
from pathlib import Path
from collections import deque
from tensorboardX import SummaryWriter

import os,sys
# print(os.getcwd())
from dqn import Agent as DQN_Agent
# from ppo.agent import Agent as PPO_Agent

import torch
import os
import numpy as np

# base_dir = Path(__file__).resolve().parent.parent
# print(base_dir)
# sys.path.append(str(base_dir))


from env.transport_dqn import Transport


# parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")
# boolean = lambda x: str(x).lower() == 'true'

# 设置环境参数
# TODO conf --> parser

map_path = 'env'
name ='Materials Transport'
conf = {
        'n_player': 1,#玩家数量
        'board_width': 11,#地图宽
        'board_height': 11,#地图高
        'n_cell_type': 5,#格子的种类
        'materials': 4,#集散点数量
        'cars': 1,#汽车数量
        'planes': 0,#飞机数量
        'barriers': 12,#固定障碍物数量
        'max_step' :2000,#最大步数
        'game_name':name,#游戏名字
        'K': 5,#每个K局更新集散点物资数目
        'map_path': 'map.txt',#存放初始地图
        'cell_range': 6,  # 单格中各维度取值范围（tuple类型，只有一个int自动转为tuple）##?
        'ob_board_width': None,  # 不同智能体观察到的网格宽度（tuple类型），None表示与实际网格相同##?
        'ob_board_height': None,  # 不同智能体观察到的网格高度（tuple类型），None表示与实际网格相同##?
        'ob_cell_range': None,  # 不同智能体观察到的单格中各维度取值范围（二维tuple类型），None表示与实际网格相同##?
    }

# Task parameters
project_root =  Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Train an agent in the transport environment")
boolean = lambda x: str(x).lower() == 'true'

parser.add_argument("--train", type=boolean, default=True, help="Whether to train the model or just evaluate it")
parser.add_argument("--load-model", default=False, action='store_true', help="Whether to load the model from the last checkpoint")
parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")

# Environment parameters
parser.add_argument("--num-agents", type=int, default= conf['n_player'], help="Number of agents in each episode")
parser.add_argument('--board-weight', type = int, default = 11)
parser.add_argument('--board-height', type = int, default = 11)

# Training parameters
parser.add_argument("--agent-type", default="dqn", choices=["dqn", "ppo"], help="Which type of RL agent to use")
parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes to train for")
parser.add_argument("--epsilon-decay", type=float, default=0.998, help="Decay factor for epsilon-greedy exploration")

parser.add_argument("--episode-length", default = conf['max_step'], type = int)

parser.add_argument("--test-id", default = 'test13')

flags = parser.parse_args()

# Seeded RNG so we can replicate our results
np.random.seed(1)

# a = os.path.exists( 'agents/'+ flags.test_id)
#
# if not os.path.exists( 'agents/'+ flags.test_id):
#     os.makedirs(flags.test_id)
# else:
#     pass

ESOURCE_PATH = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(ESOURCE_PATH , 'RzrqResFile', flags.test_id)
if not os.path.exists(path):
    os.makedirs(path)


# Create a tensorboard SummaryWriter
summary = SummaryWriter( 'runs/' + flags.test_id) # !!!runs不带.

# Helper function to generate a report
# TODO CHECK
def get_report(show_time=False):
    training = 'Training' if flags.train else 'Evaluating'
    return '  |  '.join(filter(None, [
        f'\r{training} {flags.num_agents} Agents on {flags.grid_width} x {flags.grid_height} Map',
        f'Episode {episode:<5}',
        f'Average Score: {np.mean(scores_window):.3f}',
        f'Average Steps Taken: {np.mean(steps_window):<6.1f}',
        f'Collisions: {100 * np.mean(collisions_window):>5.2f}%',
        f'Finished: {100 * np.mean(done_window):>6.2f}%',
        f'Epsilon: {eps:.2f}' if flags.agent_type == "dqn" else None,
        f'Time taken: {time.time() - start_time:.2f}s' if show_time else None])) + '  '

# Create the Transport environment
env = Transport(conf)

# render

# TODO UPDATE
# Calculate the state size based on the number of nodes in the tree observation
state_size = 122
action_size = 5

# Add some variables to keep track of the progress
scores_window, steps_window = [deque(maxlen=200) for _ in range(2)] # a, b = [deque([]), deque([])]
agent_obs = [None] * flags.num_agents # [None, None]
agent_obs_buffer = [None] * flags.num_agents
agent_action_buffer = [2] * flags.num_agents
max_steps = flags.episode_length
start_time = time.time()

# Load an RL agent and initialize it from checkpoint if necessary
# independent dqn/ppo -->每个人obs不同，同一个model
if flags.agent_type == "dqn":
    agent = DQN_Agent(state_size, action_size, flags.num_agents)
elif flags.agent_type == "ppo":
    agent = PPO_Agent(state_size, action_size, flags.num_agents)

if flags.load_model:
      start, eps = agent.load(project_root / 'checkpoints', 0, 1.0)
else: start, eps = 0, 1

if not flags.train:
    eps = 0.0

# Helper function to detect collisions
ACTIONS = {0: "up", 1: "right", 2: "down", 3: "left",4:"stop"}

def obs_wrapper(obss):
    '''
    utility: [list] -> [array]
    '''
    return np.array(obss)

def action_wrapper(action_dict):

    action = action_dict[0] # index: agent id
    each = [0] * 5
    each[action] = 1
    action_one_hot = [[each]]

    # add: 前两个action
    joint_action = []

    for i in range(2):
        player = []
        for j in range(1):
            each = [0] * 11
            # idx = np.random.randint(11)
            each[3] = 1
            player.append(each)
        joint_action.append(player)

    joint_action.append([action_one_hot[0][0]])

    return joint_action


# Main training loop
for episode in range(start + 1, flags.num_episodes + 1):
    # print('epi', episode)
    agent.reset()
    obs = env.reset()
    score, steps_taken = 0, 0

    # Build initial observations for each agent
    for a in range(flags.num_agents):
        agent_obs[a] = obs_wrapper(obs)
        agent_obs_buffer[a] = agent_obs[a].copy()

    action_dict = {}

    # Run an episode
    for step in range(flags.episode_length):
        env._render()
        steps_taken += 1
        update_values = [False] * flags.num_agents
        action_dict = {}

        # 对每一个agent算法进行更新
        # TODO UPDATE
        # for a in range(flags.num_agents):
        #     if info['action_required'][a]:
        #           action_dict[a] = agent.act(agent_obs[a], eps=eps)
        #           # action_dict[a] = np.random.randint(5)
        #           update_values[a] = True
        #           steps_taken += 1
        #     else: action_dict[a] = 0

        action_dict[a] = agent.act(agent_obs[0], eps = eps)

        joint_action = action_wrapper(action_dict)

        obs, rewards, done, info = env.step(joint_action)
        # if rewards[0][0] != 0 or rewards[0][1] != 0:

        score += rewards[0][0]
        # if score != 0:


        # Check for collisions and episode completion
        # if step == max_steps - 1:
        #     done['__all__'] = True



        # Update replay buffer and train agent
        for a in range(flags.num_agents):
            if flags.train:  #and (update_values[0] or done or done['__all__']):
                # push: agent_id, obs_old, action, reward, obs_new, done
                agent.step(a, agent_obs_buffer[a], agent_action_buffer[a], rewards[0][a], agent_obs[a], [done][a])
                agent_obs_buffer[a] = agent_obs[a].copy()
                agent_action_buffer[a] = action_dict[a]



            # if obs[a]:
            #     print('obs[a]', obs[a])
            #     print('agent_obs[a]', agent_obs[a])
            agent_obs[a] = obs_wrapper(obs)

        # Render
        # if flags.render_interval and episode % flags.render_interval == 0:
        # if collision and all(agent.position for agent in env.agents):
        #     render()
        #     print("Collisions detected by agent(s)", ', '.join(str(a) for a in obs if is_collision(a)))
        #     break

        # print(score, step, env.is_terminal())

        # agent.save(project_root + '/checkpoints/' + flags.test_id)
        # torch.save(agent.qnetwork_local.state_dict(),
        #            './' + 'test10' +  '/navigator_checkpoint'  + '.pth')

        # print(env.is_terminal() or score == 4 )


        if env.is_terminal() or score == 8 :
            # print(score, steps_taken)
            break
    if eps % 50 == 0:
        agent.save('./' + flags.test_id + '/navigator_checkpoint' + '.pth')
    # print(steps_taken)
    scores_window.append(score)
    steps_window.append(steps_taken)
    print('\r epi:{} , score:{}, step_taken:{}, score_avg:{:.3f} , step_avg:{:.3f}'.format(episode, score, steps_taken, np.mean(scores_window), np.mean(steps_window) ))
    # Epsilon decay
    if flags.train: eps = max(0.1, flags.epsilon_decay * eps)

    # Save some training statistics in their respective deques



    # scores_window.append(score )
    # steps_window.append(steps_taken)

    '''
    # Generate training reports, saving our progress every so often
    print(get_report(), end=" ")
    if episode % flags.report_interval == 0:
        print(get_report(show_time=True))
        start_time = time.time()
        if flags.train: agent.save(project_root / 'checkpoints', episode, eps)
    '''
    # Add stats to the tensorboard summary
    summary.add_scalar('performance/scores_window', np.mean(scores_window), episode)
    summary.add_scalar('performance/steps_window', np.mean(steps_window), episode)
    summary.add_scalar('performance/score', score, episode)
    summary.add_scalar('performance/steps_taken', steps_taken, episode)

print('training_time:', time.time() - start_time)
torch.save(agent.qnetwork_local.state_dict(),
                   './' + flags.test_id +  '/navigator_checkpoint'  + '.pth')