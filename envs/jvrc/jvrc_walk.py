import os
import numpy as np
import transforms3d as tf3  # 用于3D变换和欧拉角-四元数转换
import collections

from tasks import walking_task  # 行走任务定义，包含奖励函数和终止条件
from envs.common import mujoco_env  # MuJoCo环境基类
from envs.common import robot_interface  # 机器人接口，用于与MuJoCo交互
from envs.jvrc import robot  # JVRC机器人特定实现

from .gen_xml import builder  # XML模型生成器


class JvrcWalkEnv(mujoco_env.MujocoEnv):
    """JVRC机器人行走环境类，继承自MuJoCo环境基类"""

    def __init__(self):
        # 时间步长设置
        sim_dt = 0.0025  # 仿真步长：2.5毫秒
        control_dt = 0.025  # 控制步长：25毫秒
        frame_skip = (control_dt / sim_dt)  # 帧跳过：每10个仿真步执行一次控制

        # MuJoCo模型文件路径
        path_to_xml_out = '/tmp/mjcf-export/jvrc_walk/jvrc1.xml'
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)  # 如果XML文件不存在，则生成
        # 初始化MuJoCo环境
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        # PD控制器增益设置（比例-微分控制）
        # 12个关节，每个关节有P增益和D增益
        pdgains = np.zeros((12, 2))
        coeff = 0.5  # 增益系数
        # 设置P增益（比例项）
        pdgains.T[0] = coeff * np.array([200, 200, 200, 250, 80, 80,
                                         200, 200, 200, 250, 80, 80, ])
        # 设置D增益（微分项）
        pdgains.T[1] = coeff * np.array([20, 20, 20, 25, 8, 8,
                                         20, 20, 20, 25, 8, 8, ])

        # 执行器列表定义 - 对应12个腿部关节
        # 顺序：右腿 -> 左腿
        # RHIP_P, RHIP_R, RHIP_Y, RKNEE, RANKLE_R, RANKLE_P
        # LHIP_P, LHIP_R, LHIP_Y, LKNEE, LANKLE_R, LANKLE_P
        # RHIP_P - 右髋部俯仰关节（Pitch）   LHIP_P - 左髋部俯仰关节（Pitch）
        # RHIP_R - 右髋部横滚关节（Roll）    LHIP_R - 左髋部横滚关节（Roll）
        # RHIP_Y - 右髋部偏航关节（Yaw）     LHIP_Y - 左髋部偏航关节（Yaw）
        # RKNEE - 右膝关节                 LKNEE - 左膝关节
        # RANKLE_R - 右踝部横滚关节（Roll）  LANKLE_R - 左踝部横滚关节（Roll）
        # RANKLE_P - 右踝部俯仰关节（Pitch） LANKLE_P - 左踝部俯仰关节（Pitch）
        self.actuators = [0, 1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10, 11]

        # 设置机器人接口
        # 参数：模型、数据、右脚体名称、左脚体名称
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S')

        # 设置行走任务
        self.task = walking_task.WalkingTask(
            client=self.interface,  # 机器人接口
            dt=control_dt,  # 控制时间步长
            neutral_foot_orient=np.array([1, 0, 0, 0]),  # 中立脚部朝向（四元数）
            root_body='PELVIS_S',  # 根节点（骨盆）体名称
            lfoot_body='L_ANKLE_P_S',  # 左脚体名称
            rfoot_body='R_ANKLE_P_S',  # 右脚体名称
            head_body='NECK_P_S',  # 头部体名称
        )

        # 设置任务参数
        self.task._goal_height_ref = 0.80  # 目标高度参考值
        self.task._total_duration = 1.1  # 总步态周期持续时间
        self.task._swing_duration = 0.75  # 摆动相持续时间
        self.task._stance_duration = 0.35  # 支撑相持续时间

        # 重置任务状态
        self.task.reset()

        # 创建JVRC机器人实例
        self.robot = robot.JVRC(pdgains.T, control_dt, self.actuators, self.interface)
        # 设置中立姿势
        self.task._neutral_pose = self.robot.nominal_pose

        # 定义观察空间的镜像索引（用于处理左右对称性）
        base_mir_obs = [
            0.1, -1, 2, -3,  # 根节点朝向（四元数）：保持w，x取反，y保持，z取反
            -4, 5, -6,  # 根节点角速度：x取反，y保持，z取反
            13, -14, -15, 16, -17, 18,  # 电机位置[1]：右腿->左腿的映射
            7, -8, -9, 10, -11, 12,  # 电机位置[2]：左腿->右腿的映射
            25, -26, -27, 28, -29, 30,  # 电机速度[1]：右腿->左腿的映射
            19, -20, -21, 22, -23, 24,  # 电机速度[2]：左腿->右腿的映射
        ]

        # 添加额外的观察指标（时钟信号等）
        append_obs = [(len(base_mir_obs) + i) for i in range(6)]

        # 设置机器人时钟索引和镜像映射
        self.robot.clock_inds = append_obs[0:2]  # 时钟信号的索引
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()  # 镜像观察

        # 设置动作镜像映射（左右腿对称）
        self.robot.mirrored_acts = [
            6, -7, -8, 9, -10, 11,  # 右腿动作映射到左腿
            0.1, -1, -2, 3, -4, 5,  # 左腿动作映射到右腿
        ]

        # 设置动作空间
        action_space_size = len(self.robot.actuators)  # 动作空间维度=执行器数量
        action = np.zeros(action_space_size)  # 初始化动作为零
        self.action_space = np.zeros(action_space_size)  # 动作空间

        # 设置观察空间
        self.base_obs_len = 37  # 基础观察向量长度
        self.observation_space = np.zeros(self.base_obs_len)  # 观察空间

        # 重置模型
        self.reset_model()

    def get_obs(self):
        """获取当前环境的观察向量"""

        # 外部状态：时钟信号和步态模式
        # 使用正弦和余弦表示相位，提供连续的周期信号
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        # 步态模式编码
        ext_state = np.concatenate((clock, self.task.mode.encode(), [self.task.mode_ref]))

        # 内部状态：机器人本体状态
        qpos = np.copy(self.interface.get_qpos())  # 位置信息
        qvel = np.copy(self.interface.get_qvel())  # 速度信息

        # 根节点欧拉角转换（只取横滚和俯仰，忽略偏航）
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        # 重新构建四元数（偏航角设为0）
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        # 根节点角速度
        root_ang_vel = qvel[3:6]

        # 获取电机位置和速度
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        # 按执行器顺序重新排列
        motor_pos = [motor_pos[i] for i in self.actuators]
        motor_vel = [motor_vel[i] for i in self.actuators]

        # 构建机器人状态向量
        robot_state = np.concatenate([
            root_orient,  # 4维：根节点朝向（四元数）
            root_ang_vel,  # 3维：根节点角速度
            motor_pos,  # 12维：电机位置
            motor_vel,  # 12维：电机速度
        ])

        # 合并内部状态和外部状态
        state = np.concatenate([robot_state, ext_state])

        # 确保观察向量维度正确
        assert state.shape == (self.base_obs_len,)
        return state.flatten()  # 返回扁平化的观察向量

    def step(self, a):
        """执行一步环境更新"""

        # 应用动作到机器人
        applied_action = self.robot.step(a)

        # 更新任务状态并计算奖励
        self.task.step()
        # 计算各项奖励分量（扭矩奖励、动作奖励等）
        rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.prev_action, applied_action)
        # 计算总奖励
        total_reward = sum([float(i) for i in rewards.values()])

        # 检查是否终止（摔倒、超时等）
        done = self.task.done()

        # 获取新的观察
        obs = self.get_obs()

        # 返回：观察、总奖励、终止标志、详细奖励分量
        return obs, total_reward, done, rewards

    def reset_model(self):
        """重置机器人模型到初始状态"""

        '''
        # 动力学随机化（注释掉的代码，用于训练时的域随机化）
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # 执行器关节摩擦损失
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # 执行器关节阻尼
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # 执行器关节惯量
        '''

        # 状态随机化参数
        c = 0.02
        # 获取初始位置和速度
        self.init_qpos = list(self.robot.init_qpos_)
        self.init_qvel = list(self.robot.init_qvel_)
        # 添加随机噪声以增加训练多样性
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)

        # 根据任务要求修改初始状态
        # 设置根节点高度为0.81米
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        self.init_qpos[root_adr + 2] = 0.81  # 设置Z轴位置

        # 应用新的状态到仿真
        self.set_state(
            np.asarray(self.init_qpos),
            np.asarray(self.init_qvel)
        )

        # 获取初始观察并重置任务
        obs = self.get_obs()
        self.task.reset()
        return obs