#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：LearningHumanoidWalking 
@File    ：jvcr_run.py
@Author  ：Hongli Zhao
@E-mail  ：zhaohongli8711@163.com
@Date    ：2025/10/13 13:26 
@IDE     ：PyCharm 
"""
import os
import numpy as np
import transforms3d as tf3  # 用于3D变换和欧拉角-四元数转换
import collections

from tasks import running_task
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.jvrc import robot

from .gen_xml import builder  # 用于生成MuJoCo XML模型文件


class JvrcRunEnv(mujoco_env.MujocoEnv):
    def __init__(self):
        # 设置仿真参数
        sim_dt = 0.0025  # 仿真步长：2.5ms
        control_dt = 0.025  # 控制步长：25ms
        frame_skip = (control_dt / sim_dt)  # 每控制步跳过的仿真步数

        # 生成MuJoCo模型文件路径
        path_to_xml_out = '/tmp/mjcf-export/jvrc_walk/jvrc1.xml'
        if not os.path.exists(path_to_xml_out):
            builder(path_to_xml_out)  # 如果XML文件不存在，则生成

        # 初始化父类MujocoEnv
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        # 设置PD控制器增益（比例-微分控制）
        pdgains = np.zeros((12, 2))
        coeff = 0.7  # 增益系数
        # 设置比例增益(P)和微分增益(D)
        pdgains.T[0] = coeff * np.array([200, 200, 200, 250, 80, 80,  # 右腿关节P增益
                                         200, 200, 200, 250, 80, 80, ])  # 左腿关节P增益
        pdgains.T[1] = coeff * np.array([20, 20, 20, 25, 8, 8,  # 右腿关节D增益
                                         20, 20, 20, 25, 8, 8, ])  # 左腿关节D增益

        # 定义执行器索引（对应机器人关节）
        # RHIP_P, RHIP_R, RHIP_Y, RKNEE, RANKLE_R, RANKLE_P
        # LHIP_P, LHIP_R, LHIP_Y, LKNEE, LANKLE_R, LANKLE_P
        self.actuators = [0, 1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10, 11]

        # 设置机器人接口，指定左右脚和根身体的名称
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S')

        # 设置行走任务
        self.task = running_task.RuningTask(client=self.interface,
                                             dt=control_dt,
                                             neutral_foot_orient=np.array([1, 0, 0, 0]),  # 中性脚部朝向（四元数）
                                             root_body='PELVIS_S',  # 根身体（骨盆）
                                             lfoot_body='L_ANKLE_P_S',  # 左脚身体
                                             rfoot_body='R_ANKLE_P_S',  # 右脚身体
                                             head_body='NECK_P_S',  # 头部身体
                                             )

        # 设置任务参数
        self.task._goal_height_ref = 0.80  # 目标高度参考值
        # self.task._total_duration = 0.6  # 总步态周期
        # self.task._swing_duration = 0.4  # 摆动相持续时间
        # self.task._stance_duration = 0.2  # 支撑相持续时间

        self.task._total_duration = 0.25  # 总步态周期
        self.task._swing_duration = 0.2  # 摆动相持续时间
        self.task._stance_duration = 0.05  # 支撑相持续时间

        # 重置任务状态
        self.task.reset()

        # 初始化JVRC机器人对象
        self.robot = robot.JVRC(pdgains.T, control_dt, self.actuators, self.interface)
        self.task._neutral_pose = self.robot.nominal_pose  # 设置中性姿态

        # 定义观测值的镜像映射（用于对称性处理）
        base_mir_obs = [0.1, -1, 2, -3,  # 根朝向（四元数，处理对称性）
                        -4, 5, -6,  # 根角速度（处理对称性）
                        13, -14, -15, 16, -17, 18,  # 电机位置[1]（左右对称映射）
                        7, -8, -9, 10, -11, 12,  # 电机位置[2]（左右对称映射）
                        25, -26, -27, 28, -29, 30,  # 电机速度[1]（左右对称映射）
                        19, -20, -21, 22, -23, 24,  # 电机速度[2]（左右对称映射）
                        ]
        append_obs = [(len(base_mir_obs) + i) for i in range(6)]  # 附加观测索引
        self.robot.clock_inds = append_obs[0:2]  # 时钟信号索引
        self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()  # 完整镜像观测映射

        # 定义动作的镜像映射
        self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,  # 左腿动作映射到右腿
                                    0.1, -1, -2, 3, -4, 5, ]  # 右腿动作映射到左腿

        # 设置动作空间
        action_space_size = len(self.robot.actuators)  # 动作空间维度=执行器数量
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)

        # 设置观测空间
        self.base_obs_len = 37  # 基础观测维度
        self.observation_space = np.zeros(self.base_obs_len)

        # 重置模型
        self.reset_model()


    def get_obs(self):
        """获取当前环境的观测值"""
        # 外部状态：时钟信号和步态模式
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]  # 相位时钟信号
        ext_state = np.concatenate((clock, self.task.mode.encode(), [self.task.mode_ref]))  # 外部状态

        # 内部状态：机器人状态
        qpos = np.copy(self.interface.get_qpos())  # 位置状态
        qvel = np.copy(self.interface.get_qvel())  # 速度状态

        # 根身体的欧拉角转四元数（只保留横滚和俯仰，忽略偏航）
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)  # 重建四元数（偏航角设为0）
        root_ang_vel = qvel[3:6]  # 根身体角速度

        # 获取电机位置和速度
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        motor_pos = [motor_pos[i] for i in self.actuators]  # 只保留执行器对应的关节
        motor_vel = [motor_vel[i] for i in self.actuators]

        # 组合机器人状态
        robot_state = np.concatenate([
            root_orient,  # 根朝向（4维）
            root_ang_vel,  # 根角速度（3维）
            motor_pos,  # 电机位置（12维）
            motor_vel,  # 电机速度（12维）
        ])

        # 组合完整状态
        state = np.concatenate([robot_state, ext_state])  # 机器人状态 + 外部状态
        assert state.shape == (self.base_obs_len,)
        return state.flatten()

    def step(self, a):
        """执行一步动作"""
        # 机器人执行动作
        applied_action = self.robot.step(a)

        # 计算奖励
        self.task.step()  # 任务状态更新
        # 计算各项奖励（基于扭矩、动作等）
        rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.prev_action, applied_action)
        total_reward = sum([float(i) for i in rewards.values()])  # 总奖励

        # 检查是否终止
        done = self.task.done()

        # 获取新观测
        obs = self.get_obs()
        return obs, total_reward, done, rewards

    def reset_model(self):
        """重置模型到初始状态"""
        '''
        # 动力学随机化（当前被注释掉）
        # 这可以增加训练的鲁棒性，让策略适应不同的动力学参数
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # 执行器关节摩擦损失
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # 执行器关节阻尼
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # 执行器关节惯量
        '''

        # 初始状态随机化
        c = 0.02  # 随机化范围
        self.init_qpos = list(self.robot.init_qpos_)
        self.init_qvel = list(self.robot.init_qvel_)
        # 添加随机噪声到初始状态
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)

        # 根据任务修改初始状态：设置根身体高度
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        self.init_qpos[root_adr + 2] = 0.81  # 设置Z轴高度

        # 设置仿真状态
        self.set_state(
            np.asarray(self.init_qpos),
            np.asarray(self.init_qvel)
        )

        # 获取初始观测并重置任务
        obs = self.get_obs()
        self.task.reset()
        return obs