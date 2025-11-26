import numpy as np


class JVRC:
    def __init__(self, pdgains, dt, active, client):

        self.client = client
        self.control_dt = dt

        # list of desired actuators
        self.actuators = active

        # set PD gains
        self.kp = pdgains[0]
        self.kd = pdgains[1]
        assert self.kp.shape == self.kd.shape == (self.client.nu(),)
        # assert self.kp.shape == self.kd.shape
        self.client.set_pd_gains(self.kp, self.kd)

        # define init qpos and qvel
        self.init_qpos_ = [0] * self.client.nq()
        self.init_qvel_ = [0] * self.client.nv()

        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # frame skip parameter
        if (np.around(self.control_dt % self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt / self.client.sim_dt())

        # define nominal pose - 扩展为包含手臂的完整姿势
        base_position = [0, 0, 0.81]
        base_orientation = [1, 0, 0, 0]

        # 扩展为26个关节的角度（12腿 + 14臂）
        # 单位：度
        half_sitting_pose = [
            # 右腿（6个关节）
            -30, 0, 0, 50, 0, -24,
            # 左腿（6个关节）
            -30, 0, 0, 50, 0, -24,
            # 右臂（7个关节）：自然下垂，稍微弯曲
            0, 15, 0,  # 右肩：俯仰、横滚、偏航 (R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y)
            -60, 0,  # 右肘：弯曲 (R_ELBOW_P, R_ELBOW_Y)
            0, 0,  # 右腕：横滚、偏航 (R_WRIST_R, R_WRIST_Y)
            # 左臂（7个关节）：自然下垂，稍微弯曲
            0, 15, 0,  # 左肩：俯仰、横滚、偏航 (L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y)
            -60, 0,  # 左肘：弯曲 (L_ELBOW_P, L_ELBOW_Y)
            0, 0,  # 左腕：横滚、偏航 (L_WRIST_R, L_WRIST_Y)
        ]

        # 验证姿势维度是否正确
        expected_joints = 26  # 12腿 + 14臂
        if len(half_sitting_pose) != expected_joints:
            raise ValueError(f"标称姿势应该有{expected_joints}个关节，但提供了{len(half_sitting_pose)}个")

        # number of all joints
        self.num_joints = len(half_sitting_pose)

        # 转换为弧度
        self.nominal_pose = [q * np.pi / 180.0 for q in half_sitting_pose]

        # 构建完整的机器人姿势：基座位置 + 基座朝向 + 关节姿势
        # 注意：这里假设前7个是基座（3位置 + 4朝向），后面是关节
        robot_pose = base_position + base_orientation + self.nominal_pose

        # 验证姿势维度是否与模型匹配
        if len(robot_pose) != self.client.nq():
            # 如果不匹配，可能是模型包含更多关节，我们只设置我们控制的部分
            print(f"警告：机器人姿势维度({len(robot_pose)})与模型nq({self.client.nq()})不匹配")
            # 只设置我们能设置的部分
            min_len = min(len(robot_pose), self.client.nq())
            self.init_qpos_[:min_len] = robot_pose[:min_len]
        else:
            self.init_qpos_ = robot_pose

        # 定义执行器关节的标称姿势
        motor_qposadr = self.client.get_motor_qposadr()
        self.motor_offset = [self.init_qpos_[i] for i in motor_qposadr]

        # 验证执行器数量是否匹配
        if len(self.actuators) != expected_joints:
            print(f"警告：执行器数量({len(self.actuators)})与预期({expected_joints})不匹配")

    def step(self, action):
        # 创建过滤后的动作数组，大小为所有执行器数量
        filtered_action = np.zeros(len(self.motor_offset))

        # 将动作应用到选择的执行器上
        for idx, act_id in enumerate(self.actuators):
            if act_id < len(filtered_action):
                filtered_action[act_id] = action[idx]
            else:
                print(f"警告：执行器索引{act_id}超出范围(0-{len(filtered_action) - 1})")

        # 添加固定的电机偏移（标称姿势）
        filtered_action += self.motor_offset

        # 初始化之前的动作和扭矩（如果是第一步）
        if self.prev_action is None:
            self.prev_action = filtered_action.copy()
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        # 设置PD增益并执行仿真
        self.client.set_pd_gains(self.kp, self.kd)
        self.do_simulation(filtered_action, self.frame_skip)

        # 更新之前的状态
        self.prev_action = filtered_action.copy()
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        return filtered_action

    def do_simulation(self, target, n_frames):
        """执行多步仿真"""
        ratio = self.client.get_gear_ratios()
        for _ in range(n_frames):
            # 计算PD控制扭矩
            tau = self.client.step_pd(target, np.zeros(self.client.nu()))
            # 考虑齿轮比调整扭矩
            tau = [(i / j) for i, j in zip(tau, ratio)]
            # 设置电机扭矩并步进仿真
            self.client.set_motor_torque(tau)
            self.client.step()

    def get_nominal_pose_for_actuators(self):
        """获取仅针对激活执行器的标称姿势"""
        nominal_for_actuators = []
        for act_id in self.actuators:
            if act_id < len(self.motor_offset):
                nominal_for_actuators.append(self.motor_offset[act_id])
            else:
                nominal_for_actuators.append(0.0)
        return np.array(nominal_for_actuators)