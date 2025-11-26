import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.base import Net, normc_fn  # 基础网络类和初始化函数


# 评论家（值函数）网络基类，包含奖励和状态归一化功能（可选）
class Critic(Net):
    def __init__(self):
        super(Critic, self).__init__()

        # Welford算法参数，用于在线计算奖励的均值和方差
        self.welford_reward_mean = 0.0  # 奖励均值
        self.welford_reward_mean_diff = 1.0  # 奖励方差相关量
        self.welford_reward_n = 1  # 样本计数

    def forward(self):
        raise NotImplementedError

    def normalize_reward(self, r, update=True):
        """使用Welford算法归一化奖励

    Welford算法可以在线计算均值和方差，适用于流式数据
    公式：
      n += 1
      delta = r - mean
      mean += delta / n
      delta2 = r - new_mean
      M2 += delta * delta2

    参数:
      r: 奖励值
      update: 是否更新统计量
    """
        if update:
            # 处理不同维度的输入
            if len(r.size()) == 1:  # 一维张量（单个奖励）
                r_old = self.welford_reward_mean
                # 更新均值
                self.welford_reward_mean += (r - r_old) / self.welford_reward_n
                # 更新方差相关量
                self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
                self.welford_reward_n += 1
            elif len(r.size()) == 2:  # 二维张量（奖励批次）
                for r_n in r:
                    r_old = self.welford_reward_mean
                    self.welford_reward_mean += (r_n - r_old) / self.welford_reward_n
                    self.welford_reward_mean_diff += (r_n - r_old) * (r_n - r_old)
                    self.welford_reward_n += 1
            else:
                raise NotImplementedError

        # 返回标准化后的奖励：(r - mean) / std
        return (r - self.welford_reward_mean) / torch.sqrt(self.welford_reward_mean_diff / self.welford_reward_n)


class FF_V(Critic):
    """前馈状态值函数网络 - 估计V(s)"""

    def __init__(self, state_dim, layers=(256, 256), env_name='NOT SET', nonlinearity=torch.nn.functional.relu,
                 normc_init=True, obs_std=None, obs_mean=None):
        super(FF_V, self).__init__()

        # 构建多层前馈网络
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim, layers[0])]  # 输入层
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.Linear(layers[i], layers[i + 1])]  # 隐藏层
        self.network_out = nn.Linear(layers[-1], 1)  # 输出层（标量值）

        self.env_name = env_name

        self.nonlinearity = nonlinearity  # 激活函数

        # 观察值归一化参数
        self.obs_std = obs_std
        self.obs_mean = obs_mean

        # PPO论文实验中使用的权重初始化方案
        self.normc_init = normc_init

        self.init_parameters()
        self.train()  # 设置为训练模式

    def init_parameters(self):
        """初始化网络参数"""
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)  # 应用列归一化初始化

    def forward(self, inputs):
        """前向传播，估计状态值V(s)"""
        # 在非训练模式下进行观察值归一化
        if self.training == False:
            inputs = (inputs - self.obs_mean) / self.obs_std

        x = inputs
        # 前向传播通过所有层
        for l in self.critic_layers:
            x = self.nonlinearity(l(x))
        value = self.network_out(x)  # 输出状态值

        return value

    def act(self, inputs):  # 不需要，已弃用
        return self(inputs)


class FF_Q(Critic):
    """前馈动作值函数网络 - 估计Q(s,a)"""

    def __init__(self, state_dim, action_dim, layers=(256, 256), env_name='NOT SET', normc_init=True, obs_std=None,
                 obs_mean=None):
        super(FF_Q, self).__init__()

        # 网络结构：输入为状态和动作的拼接
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim + action_dim, layers[0])]  # 输入层
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.Linear(layers[i], layers[i + 1])]  # 隐藏层
        self.network_out = nn.Linear(layers[-1], 1)  # 输出层

        self.env_name = env_name

        # 观察值归一化参数
        self.obs_std = obs_std
        self.obs_mean = obs_mean

        # 权重初始化方案
        self.normc_init = normc_init

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

    def forward(self, state, action):
        """前向传播，估计动作值Q(s,a)"""
        # 非训练模式下归一化状态
        if self.training == False:
            state = (state - self.obs_mean) / self.obs_std

        # 拼接状态和动作
        x = torch.cat([state, action], len(state.size()) - 1)

        # 前向传播
        for l in self.critic_layers:
            x = F.relu(l(x))  # 使用ReLU激活函数
        value = self.network_out(x)

        return value


class Dual_Q_Critic(Critic):
    """双Q网络 - 用于减少Q学习中的过高估计问题"""

    def __init__(self, state_dim, action_dim, hidden_size=256, hidden_layers=2, env_name='NOT SET'):
        super(Dual_Q_Critic, self).__init__()

        # Q1网络结构
        self.q1_layers = nn.ModuleList()
        self.q1_layers += [nn.Linear(state_dim + action_dim, hidden_size)]
        for _ in range(hidden_layers - 1):
            self.q1_layers += [nn.Linear(hidden_size, hidden_size)]
        self.q1_out = nn.Linear(hidden_size, 1)

        # Q2网络结构（独立但结构相同）
        self.q2_layers = nn.ModuleList()
        self.q2_layers += [nn.Linear(state_dim + action_dim, hidden_size)]
        for _ in range(hidden_layers - 1):
            self.q2_layers += [nn.Linear(hidden_size, hidden_size)]
        self.q2_out = nn.Linear(hidden_size, 1)

        self.env_name = env_name

    def forward(self, state, action):
        """同时计算Q1和Q2值"""
        # 拼接状态和动作
        x1 = torch.cat([state, action], len(state.size()) - 1)
        x2 = x1  # 两个网络使用相同的输入

        # Q1前向传播
        for idx, layer in enumerate(self.q1_layers):
            x1 = F.relu(layer(x1))

        # Q2前向传播
        for idx, layer in enumerate(self.q2_layers):
            x2 = F.relu(layer(x2))

        return self.q1_out(x1), self.q2_out(x2)

    def Q1(self, state, action):
        """只计算Q1的值（常用于SAC算法）"""
        # 处理不同维度的输入
        if len(state.size()) > 2:  # 三维：序列批次
            x1 = torch.cat([state, action], 2)
        elif len(state.size()) > 1:  # 二维：批次
            x1 = torch.cat([state, action], 1)
        else:  # 一维：单个样本
            x1 = torch.cat([state, action])

        # Q1前向传播
        for idx, layer in enumerate(self.q1_layers):
            x1 = F.relu(layer(x1))

        return self.q1_out(x1)


class LSTM_Q(Critic):
    """LSTM动作值函数网络 - 用于序列数据的Q值估计"""

    def __init__(self, input_dim, action_dim, layers=(128, 128), env_name='NOT SET', normc_init=True):
        super(LSTM_Q, self).__init__()

        # LSTM网络结构
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.LSTMCell(input_dim + action_dim, layers[0])]  # 输入层
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.LSTMCell(layers[i], layers[i + 1])]  # 隐藏层
        self.network_out = nn.Linear(layers[-1], 1)  # 输出层

        self.init_hidden_state()  # 初始化隐藏状态

        self.is_recurrent = True  # 标记为循环网络
        self.env_name = env_name

        if normc_init:
            self.initialize_parameters()

    def get_hidden_state(self):
        """获取当前隐藏状态"""
        return self.hidden, self.cells

    def init_hidden_state(self, batch_size=1):
        """初始化LSTM隐藏状态为零"""
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]

    def forward(self, state, action):
        """前向传播，支持序列输入"""
        # 非训练模式下归一化（注释掉的代码）
        # if self.training == False:
        #     inputs = (inputs - self.obs_mean) / self.obs_std

        dims = len(state.size())

        # 检查状态和动作维度是否匹配
        if len(state.size()) != len(action.size()):
            print("state and action must have same number of dimensions: {} vs {}", state.size(), action.size())
            exit(1)

        # 处理轨迹批次输入（三维：序列长度 × 批次大小 × 特征维度）
        if dims == 3:
            self.init_hidden_state(batch_size=state.size(1))
            value = []
            # 按时间步处理序列
            for t, (state_batch_t, action_batch_t) in enumerate(zip(state, action)):
                # 拼接状态和动作
                x_t = torch.cat([state_batch_t, action_batch_t], 1)

                # LSTM前向传播
                for idx, layer in enumerate(self.critic_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                x_t = self.network_out(x_t)  # 输出Q值
                value.append(x_t)

            # 堆叠所有时间步的输出
            x = torch.stack([a.float() for a in value])

        else:
            # 处理单步输入
            # 注意：这里有变量名错误，应该是state而不是state_t
            x = torch.cat([state, action], len(state.size()) - 1)
            if dims == 1:  # 单个样本
                x = x.view(1, -1)

            # LSTM前向传播
            for idx, layer in enumerate(self.critic_layers):
                c, h = self.cells[idx], self.hidden[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))  # 注意：这里应该是x而不是x_t
                x = self.hidden[idx]
            x = self.network_out(x)

            if dims == 1:
                x = x.view(-1)

        return x


class LSTM_V(Critic):
    """LSTM状态值函数网络 - 用于序列数据的V值估计"""

    def __init__(self, input_dim, layers=(128, 128), env_name='NOT SET', normc_init=True):
        super(LSTM_V, self).__init__()

        # LSTM网络结构
        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.LSTMCell(input_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.critic_layers += [nn.LSTMCell(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[-1], 1)

        self.init_hidden_state()

        self.is_recurrent = True
        self.env_name = env_name

        if normc_init:
            self.initialize_parameters()

    def get_hidden_state(self):
        return self.hidden, self.cells

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.critic_layers]

    def forward(self, state):
        """前向传播，估计序列状态值V(s)"""
        # 非训练模式下归一化（注释掉的代码）
        # if self.training == False:
        #     inputs = (inputs - self.obs_mean) / self.obs_std

        dims = len(state.size())

        # 处理轨迹批次输入
        if dims == 3:
            self.init_hidden_state(batch_size=state.size(1))
            value = []
            # 按时间步处理
            for t, state_batch_t in enumerate(state):
                x_t = state_batch_t
                # LSTM前向传播
                for idx, layer in enumerate(self.critic_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                x_t = self.network_out(x_t)  # 输出V值
                value.append(x_t)

            x = torch.stack([a.float() for a in value])

        else:
            # 处理单步输入
            x = state
            if dims == 1:
                x = x.view(1, -1)

            # LSTM前向传播
            for idx, layer in enumerate(self.critic_layers):
                c, h = self.cells[idx], self.hidden[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.network_out(x)

            if dims == 1:
                x = x.view(-1)

        return x


# 类型别名
GaussianMLP_Critic = FF_V