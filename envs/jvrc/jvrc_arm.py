import os
import numpy as np
import transforms3d as tf3  # 用于3D变换和欧拉角-四元数转换
import collections

from tasks import arm_task  # 行走任务定义，包含奖励函数和终止条件
from envs.common import mujoco_env  # MuJoCo环境基类
from envs.common import robot_interface  # 机器人接口，用于与MuJoCo交互
from envs.jvrc import robot_arm as robot  # JVRC机器人特定实现

from .gen_arm_xml import builder  # XML模型生成器


class JvrcArmEnv(mujoco_env.MujocoEnv):
    """JVRC机器人行走环境类，继承自MuJoCo环境基类"""

    def __init__(self):
        # 时间步长设置
        sim_dt = 0.0025  # 仿真步长：2.5毫秒
        control_dt = 0.025  # 控制步长：25毫秒
        frame_skip = (control_dt / sim_dt)  # 帧跳过：每10个仿真步执行一次控制

        # MuJoCo模型文件路径
        path_to_xml_out = '/tmp/mjcf-export/jvrc_arm/jvrc1.xml'
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)  # 如果XML文件不存在，则生成
        # 初始化MuJoCo环境
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        # PD控制器增益设置（比例-微分控制）
        # 扩展PD增益：12个腿部关节 + 14个手臂关节 = 26个关节
        pdgains = np.zeros((26, 2))  # 从12改为26
        coeff = 0.5
        # 腿部PD增益（保持不变）
        leg_p_gains = [200, 200, 200, 250, 80, 80,  # 右腿
                       200, 200, 200, 250, 80, 80]  # 左腿
        leg_d_gains = [20, 20, 20, 25, 8, 8,  # 右腿
                       20, 20, 20, 25, 8, 8]  # 左腿

        # 手臂PD增益（新增）- 比腿部增益小，因为手臂不需要那么强的控制
        arm_p_gains = [150, 100, 120, 80, 60, 40, 30,  # 右臂：肩部3 + 肘部2 + 腕部2
                       150, 100, 120, 80, 60, 40, 30]  # 左臂：肩部3 + 肘部2 + 腕部2
        arm_d_gains = [15, 10, 12, 8, 6, 4, 3,  # 右臂
                       15, 10, 12, 8, 6, 4, 3]  # 左臂

        pdgains.T[0] = coeff * np.array(leg_p_gains + arm_p_gains)
        pdgains.T[1] = coeff * np.array(leg_d_gains + arm_d_gains)

        # 执行器列表定义 - 对应12个腿部关节  根据models/jvrc_mj_description/xml/jvrc1.xml文件进行映射
        # 顺序：右腿 -> 左腿
        # RHIP_P, RHIP_R, RHIP_Y, RKNEE, RANKLE_R, RANKLE_P
        # LHIP_P, LHIP_R, LHIP_Y, LKNEE, LANKLE_R, LANKLE_P
        # RHIP_P - 右髋部俯仰关节（Pitch）   LHIP_P - 左髋部俯仰关节（Pitch）
        # RHIP_R - 右髋部横滚关节（Roll）    LHIP_R - 左髋部横滚关节（Roll）
        # RHIP_Y - 右髋部偏航关节（Yaw）     LHIP_Y - 左髋部偏航关节（Yaw）
        # RKNEE - 右膝关节                 LKNEE - 左膝关节
        # RANKLE_R - 右踝部横滚关节（Roll）  LANKLE_R - 左踝部横滚关节（Roll）
        # RANKLE_P - 右踝部俯仰关节（Pitch） LANKLE_P - 左踝部俯仰关节（Pitch）
        # self.actuators = [
        #     # 腿部关节（0-11）
        #     0, 1, 2, 3, 4, 5,  # 右腿
        #     6, 7, 8, 9, 10, 11,  # 左腿
        #     # 手臂关节（18-31）- 跳过拇指控制，因为手指通常不需要主动控制
        #     18, 19, 20, 21, 22, 23, 24,  # 右臂：R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y,
        #     # R_ELBOW_P, R_ELBOW_Y, R_WRIST_R, R_WRIST_Y
        #     25, 26, 27, 28, 29, 30, 31  # 左臂：L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y,
        #     # L_ELBOW_P, L_ELBOW_Y, L_WRIST_R, L_WRIST_Y
        # ]
        self.actuators = [
            # 腿部关节（0-11）
            0, 1, 2, 3, 4, 5,  # 右腿
            6, 7, 8, 9, 10, 11,  # 左腿
            # 手臂关节（18-31）- 跳过拇指控制，因为手指通常不需要主动控制
            12, 13, 14, 15, 16, 17, 18,  # 右臂：R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y,
            # R_ELBOW_P, R_ELBOW_Y, R_WRIST_R, R_WRIST_Y
            19, 20, 21, 22, 23, 24, 25  # 左臂：L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y,
            # L_ELBOW_P, L_ELBOW_Y, L_WRIST_R, L_WRIST_Y
        ]

        # 设置机器人接口
        # 参数：模型、数据、右脚体名称、左脚体名称
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S')

        # 设置行走任务
        self.task = arm_task.ArmTask(
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

        # 定义新的镜像映射，包含手臂关节
        base_mir_obs = [
            # 根节点朝向和角速度（保持不变）
            0.1, -1, 2, -3,  # root orient
            -4, 5, -6,  # root ang vel
            # 腿部电机位置（保持不变）
            13, -14, -15, 16, -17, 18,  # motor pos [右腿]
            7, -8, -9, 10, -11, 12,  # motor pos [左腿]
            # 腿部电机速度（保持不变）
            25, -26, -27, 28, -29, 30,  # motor vel [右腿]
            19, -20, -21, 22, -23, 24,  # motor vel [左腿]
            # 手臂电机位置（新增）
            33, -34, -35, 36, -37, -38, 39,  # 右臂位置映射到左臂 (32-38 -> 39-45)
            26, -27, -28, 29, -30, -31, 32,  # 左臂位置映射到右臂 (25-31 -> 26-32)
            # 手臂电机速度（新增）
            47, -48, -49, 50, -51, -52, 53,  # 右臂速度映射到左臂 (46-52 -> 53-59)
            40, -41, -42, 43, -44, -45, 46,  # 左臂速度映射到右臂 (39-45 -> 40-46)
        ]

        # # 添加额外的观察指标（时钟信号等）
        # append_obs = [(len(base_mir_obs) + i) for i in range(6)]
        # # 设置机器人时钟索引和镜像映射
        # self.robot.clock_inds = append_obs[0:2]  # 时钟信号的索引
        # self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()  # 镜像观察

        append_obs = [(len(base_mir_obs) + i) for i in range(6)]
        self.robot.clock_inds = append_obs[0:2]
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()

        # 更新动作镜像映射
        self.robot.mirrored_acts = [
            # 腿部动作（保持不变）
            6, -7, -8, 9, -10, 11,  # 右腿映射到左腿
            0.1, -1, -2, 3, -4, 5,  # 左腿映射到右腿
            # 手臂动作（新增）
            19, -20, -21, 22, -23, -24, 25,  # 右臂映射到左臂
            12, -13, -14, 15, -16, -17, 18,  # 左臂映射到右臂
        ]

        # 设置动作空间
        action_space_size = len(self.robot.actuators)  # 动作空间维度=执行器数量
        action = np.zeros(action_space_size)  # 初始化动作为零
        self.action_space = np.zeros(action_space_size)  # 动作空间

        # # 设置观察空间
        # self.base_obs_len = 37  # 基础观察向量长度
        # self.observation_space = np.zeros(self.base_obs_len)  # 观察空间

        # 更新观测空间维度：4(朝向) + 3(角速度) + 26(位置) + 26(速度) + 6(外部状态) = 65维
        self.base_obs_len = 65
        self.observation_space = np.zeros(self.base_obs_len)

        # 重置模型
        self.reset_model()

    def get_obs(self):
        # 外部状态（保持不变）
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock, self.task.mode.encode(), [self.task.mode_ref]))

        # 内部状态
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())

        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        root_ang_vel = qvel[3:6]

        # 获取所有关节（包括手臂）的位置和速度
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        motor_pos = [motor_pos[i] for i in self.actuators]  # 现在包含26个关节
        motor_vel = [motor_vel[i] for i in self.actuators]  # 现在包含26个关节

        robot_state = np.concatenate([
            root_orient,  # 4维
            root_ang_vel,  # 3维
            motor_pos,  # 26维（12腿 + 14臂）
            motor_vel,  # 26维（12腿 + 14臂）
        ])

        state = np.concatenate([robot_state, ext_state])
        assert state.shape == (self.base_obs_len,)  # 现在应该是65维
        return state.flatten()

    # def get_obs(self):
    #     """获取当前环境的观察向量"""
    #
    #     # 外部状态：时钟信号和步态模式
    #     # 使用正弦和余弦表示相位，提供连续的周期信号
    #     clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
    #              np.cos(2 * np.pi * self.task._phase / self.task._period)]
    #     # 步态模式编码
    #     ext_state = np.concatenate((clock, self.task.mode.encode(), [self.task.mode_ref]))
    #
    #     # 内部状态：机器人本体状态
    #     qpos = np.copy(self.interface.get_qpos())  # 位置信息
    #     qvel = np.copy(self.interface.get_qvel())  # 速度信息
    #
    #     # 根节点欧拉角转换（只取横滚和俯仰，忽略偏航）
    #     root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
    #     # 重新构建四元数（偏航角设为0）
    #     root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
    #     # 根节点角速度
    #     root_ang_vel = qvel[3:6]
    #
    #     # 获取电机位置和速度
    #     motor_pos = self.interface.get_act_joint_positions()
    #     motor_vel = self.interface.get_act_joint_velocities()
    #     # 按执行器顺序重新排列
    #     motor_pos = [motor_pos[i] for i in self.actuators]
    #     motor_vel = [motor_vel[i] for i in self.actuators]
    #
    #     # 构建机器人状态向量
    #     robot_state = np.concatenate([
    #         root_orient,  # 4维：根节点朝向（四元数）
    #         root_ang_vel,  # 3维：根节点角速度
    #         motor_pos,  # 12维：电机位置
    #         motor_vel,  # 12维：电机速度
    #     ])
    #
    #     # 合并内部状态和外部状态
    #     state = np.concatenate([robot_state, ext_state])
    #
    #     # 确保观察向量维度正确
    #     assert state.shape == (self.base_obs_len,)
    #     return state.flatten()  # 返回扁平化的观察向量

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