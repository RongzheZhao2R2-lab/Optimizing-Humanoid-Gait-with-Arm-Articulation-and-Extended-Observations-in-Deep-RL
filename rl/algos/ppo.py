"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import ray
from rl.envs import WrapEnv


class PPOBuffer:
    """
    PPO经验回放缓冲区，用于存储轨迹数据并计算策略和值函数更新所需的回报
    这个容器在设计上没有针对内存分配速度进行优化，因为在策略梯度方法中
    这几乎从来不是性能瓶颈

    另一方面，经验缓冲区是策略梯度实现中经常出现差一错误和其他bug的地方，
    因此这段代码优先考虑清晰性和可读性，而不是（非常）轻微的速度优势
    （过早优化是万恶之源）
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        # 存储轨迹数据
        self.states  = []   # 状态序列
        self.actions = []   # 动作序列
        self.rewards = []   # 奖励序列
        self.values  = []   # 值函数估计序列
        self.returns = []   # 回报序列（用于训练critic）

        # 用于记录和日志的统计信息
        self.ep_returns = [] # 每个episode的总回报
        self.ep_lens    = [] # 每个episode的长度

        # 折扣因子和GAE参数
        self.gamma, self.lam = gamma, lam

        # 指针和轨迹索引
        self.ptr = 0           # 当前缓冲区位置
        self.traj_idx = [0]    # 轨迹开始位置的索引列表

    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        将一个时间步的智能体-环境交互数据添加到缓冲区
        """
        # TODO: 确保这些维度确实合理
        self.states  += [state.squeeze(0)]   # 移除批次维度
        self.actions += [action.squeeze(0)]  # 移除批次维度
        self.rewards += [reward.squeeze(0)]  # 移除批次维度
        self.values  += [value.squeeze(0)]   # 移除批次维度

        self.ptr += 1

    def finish_path(self, last_val=None):
        """完成一个轨迹的处理，计算回报"""
        # 记录当前轨迹结束位置
        self.traj_idx += [self.ptr]
        # 获取当前轨迹的奖励序列
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        # 使用最后一个状态的值函数估计作为bootstrap值
        R = last_val.squeeze(0).copy()  # 避免复制？
        # 反向计算折扣回报
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R)  # TODO: 在开头插入，保持顺序
                                  # 技术上这是O(k^2)，可能值得反转列表
                                  # BUG? 这里通过引用添加了R的副本？

        self.returns += returns

        # 记录episode统计信息
        self.ep_returns += [np.sum(rewards)]  # 总回报
        self.ep_lens    += [len(rewards)]     # 轨迹长度

    def get(self):
        """获取所有缓冲区数据"""
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class PPO:
    """PPO算法主类"""
    def __init__(self, args, save_path):
        # 超参数设置
        self.gamma          = args['gamma']           # 折扣因子
        self.lam            = args['lam']             # GAE参数
        self.lr             = args['lr']              # 学习率
        self.eps            = args['eps']             # Adam优化器的epsilon
        self.ent_coeff      = args['entropy_coeff']   # 熵系数
        self.clip           = args['clip']            # PPO裁剪参数
        self.minibatch_size = args['minibatch_size']  # 小批次大小
        self.epochs         = args['epochs']          # 每个批次的训练轮数
        self.max_traj_len   = args['max_traj_len']    # 最大轨迹长度
        self.use_gae        = args['use_gae']         # 是否使用GAE
        self.n_proc         = args['num_procs']       # 并行环境数量
        self.grad_clip      = args['max_grad_norm']   # 梯度裁剪阈值
        self.mirror_coeff   = args['mirror_coeff']    # 镜像对称损失系数
        self.eval_freq      = args['eval_freq']       # 评估频率

        self.recurrent = False  # 是否使用循环网络

        # 批次大小取决于并行环境数量
        self.batch_size = self.n_proc * self.max_traj_len

        # 值函数损失系数
        self.vf_coeff = 0.5
        # KL散度目标值，默认为None表示没有限制
        self.target_kl = None

        # 训练统计
        self.total_steps = 0
        self.highest_reward = -1
        self.limit_cores = 0

        # 训练迭代计数器
        self.iteration_count = 0

        # 保存路径和日志文件
        self.save_path = save_path
        self.eval_fn = os.path.join(self.save_path, 'eval.txt')
        with open(self.eval_fn, 'w') as out:
            out.write("test_ep_returns,test_ep_lens\n")

        self.train_fn = os.path.join(self.save_path, 'train.txt')
        with open(self.train_fn, 'w') as out:
            out.write("ep_returns,ep_lens\n")

        # Ray并行计算初始化（注释掉的代码）
        # os.environ['OMP_NUM_THREA DS'] = '1'
        # if args['redis_address'] is not None:
        #     ray.init(num_cpos=self.n_proc, redis_address=args['redis_address'])
        # else:
        #     ray.init(num_cpus=self.n_proc)

    def save(self, policy, critic, suffix=""):
        """保存策略和值函数模型"""
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch模型
        torch.save(policy, os.path.join(self.save_path, "actor" + suffix + filetype))
        torch.save(critic, os.path.join(self.save_path, "critic" + suffix + filetype))

    @ray.remote
    @torch.no_grad()
    def sample(self, env_fn, policy, critic, max_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        采样max_steps个总时间步，如果轨迹超过max_traj_len个时间步则截断

        参数:
            deterministic: 是否使用确定性策略
            anneal: 动作噪声退火系数
            term_thresh: 提前终止阈值
        """
        # 限制PyTorch使用单核心，避免与Ray worker冲突
        torch.set_num_threads(1)

        env = WrapEnv(env_fn)  # TODO
        env.robot.iteration_count = self.iteration_count

        memory = PPOBuffer(self.gamma, self.lam)
        memory_full = False

        while not memory_full:
            state = torch.Tensor(env.reset())
            done = False
            traj_len = 0

            # 初始化循环网络的隐藏状态（如果有）
            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

            # 运行一个完整的episode或直到达到最大步数
            while not done and traj_len < max_traj_len:
                # 选择动作
                action = policy(state, deterministic=deterministic, anneal=anneal)
                # 估计状态值
                value = critic(state)

                # 执行动作
                next_state, reward, done, _ = env.step(action.numpy())

                # 存储经验
                memory.store(state.numpy(), action.numpy(), reward, value.numpy())
                memory_full = (len(memory) == max_steps)

                state = torch.Tensor(next_state)
                traj_len += 1

                if memory_full:
                    break

            # 处理轨迹结束
            value = critic(state)
            # 如果轨迹没有自然结束，使用最后一个状态的值函数进行bootstrap
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

    def sample_parallel(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """并行采样多个环境的经验"""

        worker = self.sample
        args = (self, env_fn, policy, critic, min_steps // self.n_proc, max_traj_len, deterministic, anneal, term_thresh)

        # 创建工作进程池，每个进程收集min_steps//n_proc步数据
        workers = [worker.remote(*args) for _ in range(self.n_proc)]
        result = ray.get(workers)

        # 合并所有worker的缓冲区
        def merge(buffers):
            merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:
                offset = len(merged)
                merged.states  += buf.states
                merged.actions += buf.actions
                merged.rewards += buf.rewards
                merged.values  += buf.values
                merged.returns += buf.returns

                merged.ep_returns += buf.ep_returns
                merged.ep_lens    += buf.ep_lens

                merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
                merged.ptr += buf.ptr

            return merged

        total_buf = merge(result)

        return total_buf

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None):
        """更新策略网络和值函数网络"""
        policy = self.policy
        critic = self.critic
        old_policy = self.old_policy

        # 计算当前值函数估计
        values = critic(obs_batch)
        # 计算当前策略的动作分布和对数概率
        pdf = policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        # 计算旧策略的对数概率
        old_pdf = old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        # 新旧策略的概率比，第一次迭代时应该为1
        ratio = (log_probs - old_log_probs).exp()

        # 裁剪的替代损失
        cpi_loss = ratio * advantage_batch * mask  # 未裁剪的损失
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask  # 裁剪后的损失
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()  # PPO目标函数

        # 仅用于记录：裁剪比例
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item()

        # 值函数损失，使用TD(gae_lambda)目标
        critic_loss = self.vf_coeff * F.mse_loss(return_batch, values)

        # 熵惩罚，鼓励探索
        entropy_penalty = -(pdf.entropy() * mask).mean()

        # 镜像对称损失（用于人形机器人等对称系统）
        if mirror_observation is not None and mirror_action is not None:
            deterministic_actions = policy(obs_batch)
            mir_obs = mirror_observation(obs_batch)
            mirror_actions = policy(mir_obs)
            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = torch.Tensor([0])

        # 计算近似的反向KL散度，用于早停
        # 参考：https://github.com/DLR-RM/stable-baselines3/issues/417
        # 和PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # 以及Schulman博客: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)

        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            mirror_loss,
            clip_fraction,
        )

    def train(self,
              env_fn,
              policy,
              critic,
              n_itr,
              anneal_rate=1.0):
        """主训练循环"""

        # 初始化旧策略（用于重要性采样）
        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic

        # 优化器
        self.actor_optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time()

        # 镜像函数（如果环境支持）
        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), 'mirror_observation'):
            obs_mirr = env_fn().mirror_clock_observation

        if hasattr(env_fn(), 'mirror_action'):
            act_mirr = env_fn().mirror_action

        # 课程学习参数
        curr_anneal = 1.0    # 当前退火系数
        curr_thresh = 0      # 当前终止阈值
        start_itr = 0        # 开始迭代
        ep_counter = 0       # episode计数器
        do_term = False      # 是否启用提前终止

        # 评估统计
        test_ep_lens = []
        test_ep_returns = []

        # 主训练循环
        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            # 设置迭代计数（可用于课程学习）
            self.iteration_count = itr

            # 采样阶段
            sample_start_time = time.time()
            # 自适应退火：当性能足够好时减少探索
            if self.highest_reward > (2/3)*self.max_traj_len and curr_anneal > 0.5:
                curr_anneal *= anneal_rate
            # 自适应提前终止阈值
            if do_term and curr_thresh < 0.35:
                curr_thresh = .1 * 1.0006**(itr-start_itr)

            # 并行采样经验
            batch = self.sample_parallel(env_fn, self.policy, self.critic, self.batch_size, self.max_traj_len, anneal=curr_anneal, term_thresh=curr_thresh)
            observations, actions, returns, values = map(torch.Tensor, batch.get())

            num_samples = batch.storage_size()
            elapsed = time.time() - sample_start_time
            print("Sampling took {:.2f}s for {} steps.".format(elapsed, num_samples))

            # 标准化优势函数
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples

            # 更新旧策略
            self.old_policy.load_state_dict(policy.state_dict())

            # 早停标志
            continue_training = True

            # 优化阶段
            optimizer_start_time = time.time()
            for epoch in range(self.epochs):
                actor_losses = []
                entropies = []
                critic_losses = []
                kls = []
                mirror_losses = []
                clip_fractions = []

                # 创建小批次采样器
                if self.recurrent:
                    # 循环网络：按轨迹采样
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    # 前馈网络：随机采样
                    random_indices = SubsetRandomSampler(range(num_samples))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                # 小批次训练
                for indices in sampler:
                    if self.recurrent:
                        # 处理可变长度序列
                        obs_batch       = [observations[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        action_batch    = [actions[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        return_batch    = [returns[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        mask            = [torch.ones_like(r) for r in return_batch]

                        # 填充序列
                        obs_batch       = pad_sequence(obs_batch, batch_first=False)
                        action_batch    = pad_sequence(action_batch, batch_first=False)
                        return_batch    = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask            = pad_sequence(mask, batch_first=False)
                    else:
                        # 前馈网络：直接索引
                        obs_batch       = observations[indices]
                        action_batch    = actions[indices]
                        return_batch    = returns[indices]
                        advantage_batch = advantages[indices]
                        mask            = 1

                    # 更新策略
                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, mirror_loss, clip_fraction = scalars

                    # 记录损失
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    clip_fractions.append(clip_fraction)

                    # KL早停检查
                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # 策略网络更新
                    self.actor_optimizer.zero_grad()
                    (actor_loss + self.mirror_coeff*mirror_loss + self.ent_coeff*entropy_penalty).backward()

                    # 梯度裁剪，防止"不幸的"小批次导致病态更新
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    self.actor_optimizer.step()

                    # 值函数网络更新
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
                    self.critic_optimizer.step()

                # 早停
                if not continue_training:
                    break

            elapsed = time.time() - optimizer_start_time
            print("Optimizer took: {:.2f}s".format(elapsed))

            # 课程学习逻辑：当平均episode长度足够长时开始计数
            if np.mean(batch.ep_lens) >= self.max_traj_len * 0.75:
                ep_counter += 1
            if do_term == False and ep_counter > 50:
                do_term = True
                start_itr = itr

            # 输出训练统计
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Return (batch)', "%8.5g" % np.mean(batch.ep_returns)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % np.mean(batch.ep_lens)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(actor_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(critic_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mirror loss', "%8.3g" % np.mean(mirror_losses)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(kls)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(entropies)) + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(clip_fractions)) + "\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            elapsed = time.time() - train_start_time
            print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f})".format(elapsed, self.total_steps, self.total_steps/elapsed))

            # 保存训练指标
            with open(self.train_fn, 'a') as out:
                out.write("{},{}\n".format(np.mean(batch.ep_returns), np.mean(batch.ep_lens)))

            # 定期评估
            if (itr+1)%self.eval_freq==0:
                # 评估阶段
                evaluate_start = time.time()
                test = self.sample_parallel(env_fn, self.policy, self.critic, self.batch_size, self.max_traj_len, deterministic=True)
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                print("====EVALUATE EPISODE====  (Return = {})".format(avg_eval_reward))

                # 保存评估指标
                with open(self.eval_fn, 'a') as out:
                    out.write("{},{}\n".format(np.mean(test.ep_returns), np.mean(test.ep_lens)))
                test_ep_lens.append(np.mean(test.ep_lens))
                test_ep_returns.append(np.mean(test.ep_returns))

                # 绘制评估曲线
                plt.clf()
                xlabel = [i*self.eval_freq for i in range(len(test_ep_lens))]
                plt.plot(xlabel, test_ep_lens, color='blue', marker='o', label='Ep lens')
                plt.plot(xlabel, test_ep_returns, color='green', marker='o', label='Returns')
                plt.xticks(np.arange(0, itr+1, step=self.eval_freq))
                plt.xlabel('Iterations')
                plt.ylabel('Returns/Episode lengths')
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(self.save_path, 'eval.svg'), bbox_inches='tight')

                # 保存策略
                self.save(policy, critic, "_" + repr(itr))

                # 如果是最佳模型，保存为actor.pt
                if self.highest_reward < avg_eval_reward:
                    self.highest_reward = avg_eval_reward
                    self.save(policy, critic)