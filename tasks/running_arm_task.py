import numpy as np
import transforms3d as tf3  # 用于3D变换
from tasks import rewards  # 奖励计算模块
from enum import Enum, auto  # 枚举类型


class WalkModes(Enum):
    """行走模式枚举类，定义三种行走模式"""
    STANDING = auto()  # 站立模式
    INPLACE = auto()  # 原地踏步模式
    FORWARD = auto()  # 前进行走模式

    def encode(self):
        """将行走模式编码为one-hot向量"""
        if self.name == 'STANDING':
            return np.array([0, 0, 1])  # 站立：[0,0,1]
        elif self.name == 'INPLACE':
            return np.array([0, 1, 0])  # 原地踏步：[0,1,0]
        elif self.name == 'FORWARD':
            return np.array([1, 0, 0])  # 前进：[1,0,0]

    def sample_ref(self):
        """为每种模式采样参考值（速度或角速度）"""
        if self.name == 'STANDING':
            return np.random.uniform(-1, 1)  # 站立：随机偏航角速度
        if self.name == 'INPLACE':
            return np.random.uniform(-0.5, 0.5)  # 原地踏步：随机偏航角速度
        if self.name == 'FORWARD':
            return 6.0  # 前进速度参考值(m/s)


class RunningArmTask(object):
    """双足机器人动态稳定行走任务类"""

    def __init__(self,
                 client=None,  # 机器人客户端接口
                 dt=0.025,  # 控制时间步长
                 neutral_foot_orient=[],  # 中立脚部朝向
                 neutral_pose=[],  # 中立姿势
                 root_body='pelvis',  # 根节点（骨盆）体名称
                 lfoot_body='lfoot',  # 左脚体名称
                 rfoot_body='rfoot',  # 右脚体名称
                 head_body='head',  # 头部体名称
                 waist_r_joint='waist_r',  # 腰部横滚关节
                 waist_p_joint='waist_p',  # 腰部俯仰关节
                 manip_hfield=False,  # 是否操作高度场
                 ):

        self._client = client  # 机器人客户端
        self._control_dt = dt  # 控制时间步长
        self._neutral_foot_orient = neutral_foot_orient  # 中立脚部朝向
        self._neutral_pose = np.array(neutral_pose)  # 中立姿势数组
        self.manip_hfield = manip_hfield  # 高度场操作标志

        self._mass = self._client.get_robot_mass()  # 获取机器人质量

        # 这些参数依赖于具体机器人，目前硬编码
        # 理想情况下应该作为__init__的参数
        self.mode_ref = []  # 模式参考值（速度/角速度）
        self._goal_height_ref = []  # 目标高度参考
        self._swing_duration = []  # 摆动相持续时间
        self._stance_duration = []  # 支撑相持续时间
        self._total_duration = []  # 总步态周期持续时间

        # 身体部位名称
        self._root_body_name = root_body  # 根节点体
        self._lfoot_body_name = lfoot_body  # 左脚体
        self._rfoot_body_name = rfoot_body  # 右脚体
        self._head_body_name = head_body  # 头部体

    def calc_reward(self, prev_torque, prev_action, action):
        """计算奖励函数，包含多个奖励分量"""

        # 获取脚部状态信息
        self.l_foot_vel = self._client.get_lfoot_body_vel(frame=1)[0]  # 左脚速度（世界坐标系）
        self.r_foot_vel = self._client.get_rfoot_body_vel(frame=1)[0]  # 右脚速度（世界坐标系）
        self.l_foot_frc = self._client.get_lfoot_grf()  # 左脚地面反作用力
        self.r_foot_frc = self._client.get_rfoot_grf()  # 右脚地面反作用力

        neck_pos = self._client.get_object_xpos_by_name("NECK_Y_S", 'OBJ_BODY')

        # 获取头部和根节点位置（只取x,y坐标）
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, 'OBJ_BODY')[0:2]
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]

        # 获取当前关节位置
        current_pose = np.array(self._client.get_act_joint_positions())

        # 获取相位时钟函数
        r_frc = self.right_clock[0]  # 右脚力相位函数
        l_frc = self.left_clock[0]  # 左脚力相位函数
        r_vel = self.right_clock[1]  # 右脚速度相位函数
        l_vel = self.left_clock[1]  # 左脚速度相位函数

        # 站立模式下调整相位函数
        if self.mode == WalkModes.STANDING:
            r_frc = (lambda _: 1)  # 始终施加力
            l_frc = (lambda _: 1)  # 始终施加力
            r_vel = (lambda _: -1)  # 速度惩罚
            l_vel = (lambda _: -1)  # 速度惩罚

        # 根据行走模式设置目标速度参考
        if self.mode == WalkModes.STANDING:
            self._goal_speed_ref = 0  # 站立：零速度
            yaw_vel_ref = 0  # 零偏航角速度
        if self.mode == WalkModes.INPLACE:
            self._goal_speed_ref = 0  # 原地踏步：零前进速度
            yaw_vel_ref = self.mode_ref  # 随机偏航角速度
        if self.mode == WalkModes.FORWARD:
            self._goal_speed_ref = self.mode_ref  # 前进：随机前进速度
            yaw_vel_ref = 0  # 零偏航角速度

        # 警告：这里假设腿部关节在前12个位置
        # 构建奖励字典，包含多个奖励分量
        reward = dict(
            foot_frc_score=0.2 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),  # 脚部接触力奖励
            foot_vel_score=0.2 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),  # 脚部速度奖励
            # root_accel=0.030 * rewards._calc_root_accel_reward(self),  # 根身体加速度奖励（平滑性）
            # height_error=0.050 * rewards._calc_height_reward(self),  # 高度误差奖励
            # com_vel_error=0.350 * rewards._calc_fwd_vel_reward(self),  # 前进速度误差奖励
            # yaw_vel_error=0.120 * rewards._calc_yaw_vel_reward(self, yaw_vel_ref),  # 偏航角速度误差奖励
            # upper_body_reward=0.050 * np.exp(-10 * np.linalg.norm(head_pos - root_pos)),  # 上半身稳定性奖励
            # posture_error=0.040 * np.exp(-np.linalg.norm(self._neutral_pose[:12] - current_pose[:12])),  # 姿态误差奖励

            feet_separation=0.2 * rewards._calc_feet_separation_reward(self),  # 维持合适的脚步间距
            heading=0.2 * rewards._calc_heading_reward(self),  # 鼓励保持前进方向

            # 腰部姿势代价 - 保持腰部在中立位置
            # waist_cost=-0.3 * sum(abs(np.array(self._client.get_act_joint_positions()[12:15]) - np.array([0, 0.15, 0]))),
            # 高度奖励 - 基于颈部高度
            height_error=0.050 * neck_pos[2] + 0.3 * (
                        self._client.get_object_xpos_by_name("PELVIS_S", 'OBJ_BODY')[2] - 0.85),
            # height_error=0.050 * rewards._calc_height_reward(self),  # 高度误差奖励
            # 速度奖励 - 鼓励高速前进（目标速度6m/s）
            vel_reward=0.2 + 0.2 * -abs(self._client.get_body_vel("PELVIS_S")[0][0] - 6.0),
            # 侧向和旋转速度惩罚
            # velocity_penalty=0.1 + 0.13 * (-abs(self._client.get_body_vel("PELVIS_S")[0][1]) - abs(self._client.get_body_vel("PELVIS_S")[0][5])),
            velocity_penalty=0.1 * rewards._calc_orient_reward(self, 'PELVIS_S'),

            # 加速度惩罚 - 减少抖动
            # acc_penalty=0.2 - 0.013 * sum(abs(neck_acc)),
            # 朝向奖励 - 保持身体和脚部正确朝向
            orient_cost=0.1 * (2 * rewards._calc_orient_reward(self, self._root_body_name) +
                               4 * rewards._calc_body_orient_reward(self, self._rfoot_body_name) +
                               4 * rewards._calc_body_orient_reward(self, self._lfoot_body_name)) / 3,

            coordination_cost = 0.05 * rewards._calc_arm_swing_coordination(self) #鼓励手臂对称性
        )

        # 注意：各奖励分量的权重和为1.0（0.225+0.225+0.05+0.05+0.15+0.15+0.05+0.05=1.0）
        return reward

    def step(self):
        """执行一步任务更新（相位推进和模式切换）"""

        # 相位递增
        self._phase += 1
        # 相位循环：达到周期后归零
        if self._phase >= self._period:
            self._phase = 0

        # 随机在INPLACE和STANDING模式间切换（仅在双支撑期）
        in_double_support = self.right_clock[0](self._phase) == 1 and self.left_clock[0](self._phase) == 1
        if np.random.randint(100) == 0 and in_double_support:
            if self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.STANDING  # 原地踏步 -> 站立
            elif self.mode == WalkModes.STANDING:
                self.mode = WalkModes.INPLACE  # 站立 -> 原地踏步
            self.mode_ref = self.mode.sample_ref()  # 重新采样参考值

        # 随机在INPLACE和FORWARD模式间切换
        if np.random.randint(200) == 0 and self.mode != WalkModes.STANDING:
            if self.mode == WalkModes.FORWARD:
                self.mode = WalkModes.INPLACE  # 前进 -> 原地踏步
            elif self.mode == WalkModes.INPLACE:
                self.mode = WalkModes.FORWARD  # 原地踏步 -> 前进
            self.mode_ref = self.mode.sample_ref()  # 重新采样参考值

        # 操作高度场（用于地形适应训练）
        if self.manip_hfield:
            if np.random.randint(200) == 0 and self.mode != WalkModes.STANDING:
                # 随机设置高度场位置和高度
                self._client.model.geom("hfield").pos[:] = [
                    np.random.uniform(-0.5, 0.5),  # x位置随机
                    np.random.uniform(-0.5, 0.5),  # y位置随机
                    np.random.uniform(-0.015, -0.035)  # z高度随机
                ]
        return

    def done(self):
        """检查终止条件"""

        # 检查自碰撞
        contact_flag = self._client.check_self_collisions()
        # 获取位置信息
        qpos = self._client.get_qpos()

        # 终止条件字典
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.4),  # 高度过低（摔倒）
            "qpos[2]_ul": (qpos[2] > 1.4),  # 高度过高（异常）
            # "contact_flag": contact_flag,  # 发生自碰撞
        }

        # 如果任一终止条件为True，则任务结束
        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        """重置任务状态"""
        # 随机选择行走模式（概率分布不同）
        # self.mode = np.random.choice(
        #     [WalkModes.STANDING, WalkModes.INPLACE, WalkModes.FORWARD],
        #     p=[0.2, 0.3, 0.5])  # 站立5%，原地15%，前进80%
        # self.mode_ref = self.mode.sample_ref()  # 采样模式参考值

        self.mode = WalkModes.FORWARD
        self.mode_ref = 6.0  # 固定高速目标

        # 创建步态相位时钟函数
        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration,  # 摆动相持续时间
            self._stance_duration,  # 支撑相持续时间
            0.1,  # 相位偏移
            "grounded",  # 时钟类型
            1 / self._control_dt  # 控制频率
        )

        # 计算完整步态周期的控制步数（左摆动+右摆动）
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        # 在初始化时随机化相位
        self._phase = np.random.randint(0, self._period)