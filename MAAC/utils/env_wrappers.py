"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import copy


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            env._render()
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):


        # results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        # obs, rews, dones, infos = map(np.array, zip(*results))
        # self.ts += 1
        # for (i, done) in enumerate(dones):
        #     if all(done):
        #         obs[i] = self.envs[i].reset()
        #         self.ts[i] = 0
        # self.actions = None



        # #####################################
        # # method 1
        # # add: state modified
        # obs, rews, dones, infos = self.envs[0].step(self.actions)
        # # obs = [obs]
        # obs_ = []  # TODO: TO CHECK
        # state = []
        # for i in range(self.envs[0].board_height):
        #     for j in range(self.envs[0].board_width):
        #         state.append(obs[i][j][0])
        #
        # for n in range(self.envs[0].cars):  # TODO: TO CHECK
        #     obs_.append(np.array(state))
        # obs = [obs_]
        #
        # # dones
        # Done = []
        # for n in range(self.envs[0].cars):
        #     Done.append(dones)
        # Done = [Done]
        # # ######################################
        # return np.array(obs), np.array(rews), np.array(Done), infos

        #####################################
        # method 2
        # add: state modified
        obss, rews, dones, infos = self.envs[0].step(self.actions)
        obs_ = []  # TODO: TO CHECK
        # obs_ = []  # TODO: TO CHECK
        # state = []
        # for i in range(self.envs[0].board_height):
        #     for j in range(self.envs[0].board_width):
        #         state.append(obs[i][j][0])

        obs = self.obs_wrapper(obss)

        # dones
        Done = []
        for n in range(self.envs[0].cars):
            Done.append(dones)
        Done = [Done]

        return np.array(obs), np.array(rews), np.array(Done), infos
        ######################################

    def reset(self):

        # results = [env.reset() for env in self.envs]
        results = self.obs_wrapper(self.envs[0].reset())
        return np.array(results)

    def close(self):
        return

    def obs_wrapper(self, o):
        obs = o
        obs_ = []
        # obs_.append(np.array(obs[0:121] + obs[121:124] + obs[127:131]))
        # obs_.append(np.array(obs[0:121] + obs[124:127] + obs[127:131]))
        for n in range(self.envs[0].cars):  # TODO: TO CHECK
            temp_obs = copy.deepcopy(obs[:122])
            temp_obs = temp_obs + obs[-4:]
            temp_obs[obs[121+3*n+1]*11 + obs[121+3*n+2]] = 5
            obs_.append(np.array(temp_obs))
            # obs_.append(np.array(obs[:121]+obs[121+3*n:124+3*n]))
        # obs_.append(np.array(obs))
        # obs_.append(np.array(obs))
        obs = [obs_]
        return obs

    def obs_wrapper2(self, o):
        obs = o
        obs_ = []
        for n in range(self.envs[0].cars):  # TODO: TO CHECK
            obs_.append(np.array(obs))
        obs = [obs_]
        return obs

    def render(self):
        self.envs[0]._render()


    # def render(self, mode='rgb_array'):
    #     # self.envs[0].render(mode)
    #
    #     return self.envs[0].render_board()
