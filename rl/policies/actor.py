import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

from rl.policies.base import Net  # 基础网络类

# 对数标准差的范围限制
LOG_STD_HI = -1.5  # 对数标准差上限
LOG_STD_LO = -20  # 对数标准差下限


class Actor(Net):
    """演员（策略）网络基类"""

    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError


class Linear_Actor(Actor):
    """线性策略网络 - 最简单的策略网络实现"""

    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(Linear_Actor, self).__init__()

        # 网络结构：状态 -> 隐藏层 -> 动作
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

        self.action_dim = action_dim

        # 初始化所有权重为零
        for p in self.parameters():
            p.data = torch.zeros(p.shape)

    def forward(self, state):
        """前向传播"""
        a = self.l1(state)
        a = self.l2(a)
        self.action = a  # 存储当前动作
        return a

    def get_action(self):
        """获取最后计算的动作"""
        return self.action


class FF_Actor(Actor):
    """前馈神经网络策略 - 多层感知机"""

    def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=F.relu, max_action=1):
        super(FF_Actor, self).__init__()

        # 构建多层网络
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]  # 输入层
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]  # 隐藏层
        self.network_out = nn.Linear(layers[-1], action_dim)  # 输出层

        self.action = None
        self.action_dim = action_dim
        self.env_name = env_name
        self.nonlinearity = nonlinearity  # 激活函数

        self.initialize_parameters()  # 初始化参数

        self.max_action = max_action  # 动作范围限制

    def forward(self, state, deterministic=True):
        """前向传播 - 输出确定性动作"""
        x = state
        # 前向传播通过所有层
        for idx, layer in enumerate(self.actor_layers):
            x = self.nonlinearity(layer(x))

        # 使用tanh将输出限制在[-1, 1]范围内
        self.action = torch.tanh(self.network_out(x))
        # 缩放到最大动作范围
        return self.action * self.max_action

    def get_action(self):
        return self.action


class LSTM_Actor(Actor):
    """LSTM循环神经网络策略 - 用于处理序列数据"""

    def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=torch.tanh, max_action=1):
        super(LSTM_Actor, self).__init__()

        # 构建LSTM层
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]  # 第一层LSTM
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i + 1])]  # 后续LSTM层
        self.network_out = nn.Linear(layers[i - 1], action_dim)  # 输出层

        self.action = None
        self.action_dim = action_dim
        self.init_hidden_state()  # 初始化隐藏状态
        self.env_name = env_name
        self.nonlinearity = nonlinearity

        self.is_recurrent = True  # 标记为循环网络

        self.max_action = max_action

    def get_hidden_state(self):
        """获取当前隐藏状态"""
        return self.hidden, self.cells

    def set_hidden_state(self, data):
        """设置隐藏状态"""
        if len(data) != 2:
            print("Got invalid hidden state data.")
            exit(1)

        self.hidden, self.cells = data

    def init_hidden_state(self, batch_size=1):
        """初始化隐藏状态为零"""
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, x, deterministic=True):
        """前向传播，支持不同维度的输入"""
        dims = len(x.size())

        # 处理轨迹批次输入 (序列长度, 批次大小, 状态维度)
        if dims == 3:
            self.init_hidden_state(batch_size=x.size(1))
            y = []
            # 按时间步处理序列
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    # LSTM前向传播
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            # 处理单步输入
            if dims == 1:  # 单个时间步
                x = x.view(1, -1)

            # 单步前向传播
            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.nonlinearity(self.network_out(x))

            if dims == 1:
                x = x.view(-1)

        self.action = self.network_out(x)
        return self.action

    def get_action(self):
        return self.action


class Gaussian_FF_Actor(Actor):
    """高斯前馈策略网络 - 输出动作的概率分布（用于PPO等随机策略算法）"""

    def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=torch.nn.functional.relu,
                 fixed_std=None, bounded=False, normc_init=True):
        super(Gaussian_FF_Actor, self).__init__()

        # 网络结构
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i + 1])]
        self.means = nn.Linear(layers[-1], action_dim)  # 均值输出层

        # 标准差处理：可学习或固定
        if fixed_std is None:
            self.log_stds = nn.Linear(layers[-1], action_dim)  # 对数标准差输出层
            self.learn_std = True  # 可学习标准差
        else:
            self.fixed_std = fixed_std  # 固定标准差
            self.learn_std = False

        self.action = None
        self.action_dim = action_dim
        self.env_name = env_name
        self.nonlinearity = nonlinearity

        # 观察值归一化参数（默认不归一化）
        self.obs_std = 1.0
        self.obs_mean = 0.0

        # 是否使用PPO论文中的权重初始化方案
        self.normc_init = normc_init

        # 是否对均值输出使用tanh限制
        self.bounded = bounded

        self.init_parameters()

    def init_parameters(self):
        """初始化网络参数"""
        if self.normc_init:
            self.apply(normc_fn)  # 应用归一化初始化
            self.means.weight.data.mul_(0.01)  # 均值层权重缩小

    def _get_dist_params(self, state):
        """获取分布参数：均值和标准差"""
        # 观察值归一化
        state = (state - self.obs_mean) / self.obs_std

        x = state
        # 前向传播
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        mean = self.means(x)

        # 可选：对均值输出使用tanh限制
        if self.bounded:
            mean = torch.tanh(mean)

        # 计算标准差
        if self.learn_std:
            # 可学习标准差（使用复杂的变换确保正值和合理范围）
            sd = (-2 + 0.5 * torch.tanh(self.log_stds(x))).exp()
        else:
            sd = self.fixed_std  # 固定标准差

        return mean, sd

    def forward(self, state, deterministic=True, anneal=1.0):
        """前向传播

        参数:
            deterministic: 是否使用确定性策略（输出均值）
            anneal: 退火系数，用于调整探索程度
        """
        mu, sd = self._get_dist_params(state)
        sd *= anneal  # 应用退火

        if not deterministic:
            # 随机策略：从正态分布中采样
            self.action = torch.distributions.Normal(mu, sd).sample()
        else:
            # 确定性策略：直接使用均值
            self.action = mu

        return self.action

    def get_action(self):
        return self.action

    def distribution(self, inputs):
        """返回动作的概率分布对象（用于PPO计算概率比）"""
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


class Gaussian_LSTM_Actor(Actor):
    """高斯LSTM策略网络 - 结合循环网络和随机策略"""

    def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=F.tanh, normc_init=False,
                 max_action=1, fixed_std=None):
        super(Gaussian_LSTM_Actor, self).__init__()

        # LSTM网络结构
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i + 1])]
        self.network_out = nn.Linear(layers[i - 1], action_dim)

        self.action = None
        self.action_dim = action_dim
        self.init_hidden_state()
        self.env_name = env_name
        self.nonlinearity = nonlinearity
        self.max_action = max_action

        # 观察值归一化参数
        self.obs_std = 1.0
        self.obs_mean = 0.0

        self.is_recurrent = True

        # 标准差处理
        if fixed_std is None:
            self.log_stds = nn.Linear(layers[-1], action_dim)
            self.learn_std = True
        else:
            self.fixed_std = fixed_std
            self.learn_std = False

        if normc_init:
            self.initialize_parameters()

        self.act = self.forward  # 别名

    def _get_dist_params(self, state):
        """获取分布参数（支持序列输入）"""
        state = (state - self.obs_mean) / self.obs_std

        dims = len(state.size())

        x = state
        # 处理序列输入
        if dims == 3:  # 轨迹批次
            self.init_hidden_state(batch_size=x.size(1))
            action = []
            y = []
            # 按时间步处理
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            # 处理单步输入
            if dims == 1:
                x = x.view(1, -1)

            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)

        # 计算均值和标准差
        mu = self.network_out(x)
        if self.learn_std:
            # 限制对数标准差范围后取指数
            sd = torch.clamp(self.log_stds(x), LOG_STD_LO, LOG_STD_HI).exp()
        else:
            sd = self.fixed_std

        return mu, sd

    def init_hidden_state(self, batch_size=1):
        """初始化LSTM隐藏状态"""
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, state, deterministic=True, anneal=1.0):
        """前向传播"""
        mu, sd = self._get_dist_params(state)
        sd *= anneal  # 探索退火

        if not deterministic:
            self.action = torch.distributions.Normal(mu, sd).sample()
        else:
            self.action = mu

        return self.action

    def distribution(self, inputs):
        """返回动作分布"""
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)

    def get_action(self):
        return self.action


# 初始化函数（来自PPO论文的初始化方案）
# 注意：这个函数名与参数名相同曾经导致了一个严重的bug
# 因为在Python中 "if <function_name>" 会评估为True...
def normc_fn(m):
    """归一化列初始化 - 确保每列的L2范数为1"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)  # 从标准正态分布初始化
        # 按列的L2范数归一化
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)  # 偏置初始化为0