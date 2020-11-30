class Q_learning:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, e_greed=0.1):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((state_dim, action_dim))

    def sample(self, state):
        """
        使用 epsilon 贪婪策略获取动作
        return: action
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else: action = self.predict(state)
        return action

    def predict(self, state):
        """ 根据输入观察值，预测输出的动作值 """
        all_actions = self.Q[state, :]
        max_action = np.max(all_actions)
        # 防止最大的 Q 值有多个，找出所有最大的 Q，然后再随机选择
        # where函数返回一个 array， 每个元素为下标
        max_action_list = np.where(all_actions == max_action)[0]
        action = np.random.choice(max_action_list)
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning 更新 Q-table 方法
        这里没有明确选择下一个动作 next_action, 而是选择 next_state 下有最大价值的动作
        所以用 np.max(self.Q[next_state, :]) 来计算 td-target
        然后更新 Q-table
        """
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.lr * (target_q - self.Q[state, action])


def run_episode(self, render=False):
    state = self.env.reset()
    while True:
        action = self.model.sample(state)
        next_state, reward, done, _ = self.env.step(action)
        # 训练 Q-learning算法
        self.model.learn(state, action, reward, next_state, done)

        state = next_state
        if render: self.env.render()
        if done: break