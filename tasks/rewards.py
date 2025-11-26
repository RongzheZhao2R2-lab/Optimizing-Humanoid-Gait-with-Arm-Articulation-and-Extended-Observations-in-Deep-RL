import numpy as np

#===========================新增手臂部分===============================================
def _calc_arm_swing_coordination(self):
    """
    计算手臂摆动协调性奖励
    鼓励手臂与腿部运动的协调性（对侧摆动模式）
    """
    # 获取腿部关节速度（假设前12个是腿部关节）
    leg_velocities = np.array(self._client.get_act_joint_velocities()[:12])

    # 获取手臂关节速度（假设接下来的14个是手臂关节）
    arm_velocities = np.array(self._client.get_act_joint_velocities()[12:26])

    # 分离左右腿和手臂
    right_leg_vel = np.mean(np.abs(leg_velocities[:6]))  # 右腿
    left_leg_vel = np.mean(np.abs(leg_velocities[6:12]))  # 左腿
    right_arm_vel = np.mean(np.abs(arm_velocities[:7]))  # 右臂
    left_arm_vel = np.mean(np.abs(arm_velocities[7:14]))  # 左臂

    # 理想的对侧摆动模式：右腿前进时左臂前进，左腿前进时右臂前进
    # 计算对侧协调性
    contralateral_coord = np.exp(-np.abs(right_leg_vel - left_arm_vel) - np.abs(left_leg_vel - right_arm_vel))

    # 计算同侧抑制（惩罚同侧运动）
    ipsilateral_penalty = np.exp(-(np.abs(right_leg_vel - right_arm_vel) + np.abs(left_leg_vel - left_arm_vel)))

    # 综合协调性奖励
    coordination_reward = 0.7 * contralateral_coord + 0.3 * ipsilateral_penalty

    return coordination_reward


def _calc_arm_energy_efficiency(self, prev_torque, action):
    """
    计算手臂能量效率奖励
    鼓励高效的手臂运动，避免过度用力
    """
    # 手臂扭矩（假设手臂关节在索引12-25）
    arm_torque = np.array(prev_torque[12:26])
    arm_action = np.array(action[12:26])

    # 计算手臂能量消耗（基于扭矩和动作）
    torque_energy = np.linalg.norm(arm_torque)
    action_energy = np.linalg.norm(arm_action)

    # 鼓励低能量消耗
    energy_reward = np.exp(-self.arm_energy_coeff * (torque_energy + action_energy))

    return energy_reward


def _calc_arm_posture_reward(self, current_pose):
    """
    计算手臂姿势奖励
    鼓励保持自然的手臂姿势
    """
    # 手臂关节的目标自然姿势（弧度）
    # 这里定义了一个相对自然的手臂姿势
    target_arm_pose = np.array([
        # 右臂：稍微弯曲的自然姿势
        0.0,  # R_SHOULDER_P
        0.3,  # R_SHOULDER_R
        0.0,  # R_SHOULDER_Y
        -1.0,  # R_ELBOW_P
        0.0,  # R_ELBOW_Y
        0.0,  # R_WRIST_R
        0.0,  # R_WRIST_Y
        # 左臂：对称姿势
        0.0,  # L_SHOULDER_P
        0.3,  # L_SHOULDER_R
        0.0,  # L_SHOULDER_Y
        -1.0,  # L_ELBOW_P
        0.0,  # L_ELBOW_Y
        0.0,  # L_WRIST_R
        0.0,  # L_WRIST_Y
    ])

    # 当前手臂姿势（假设手臂关节在索引12-25）
    current_arm_pose = np.array(current_pose[12:26])

    # 计算姿势误差
    pose_error = np.linalg.norm(current_arm_pose - target_arm_pose)

    # 姿势奖励：误差越小奖励越高
    posture_reward = np.exp(-self.arm_posture_coeff * pose_error)

    return posture_reward


def _calc_arm_symmetry_reward(self):
    """
    计算手臂对称性奖励
    鼓励左右手臂运动的对称性
    """
    # 获取手臂关节位置和速度
    arm_positions = np.array(self._client.get_act_joint_positions()[12:26])
    arm_velocities = np.array(self._client.get_act_joint_velocities()[12:26])

    # 分离左右手臂
    right_arm_pos = arm_positions[:7]
    left_arm_pos = arm_positions[7:14]
    right_arm_vel = arm_velocities[:7]
    left_arm_vel = arm_velocities[7:14]

    # 计算对称性误差（考虑镜像对称）
    # 对于某些关节，左右应该是相反的（如肩部横滚）
    pos_symmetry_error = np.linalg.norm([
        right_arm_pos[0] - left_arm_pos[0],  # 肩部俯仰应该相同
        right_arm_pos[1] + left_arm_pos[1],  # 肩部横滚应该相反
        right_arm_pos[2] - left_arm_pos[2],  # 肩部偏航应该相同
        right_arm_pos[3] - left_arm_pos[3],  # 肘部俯仰应该相同
        right_arm_pos[4] - left_arm_pos[4],  # 肘部偏航应该相同
        right_arm_pos[5] + left_arm_pos[5],  # 腕部横滚应该相反
        right_arm_pos[6] - left_arm_pos[6],  # 腕部偏航应该相同
    ])

    vel_symmetry_error = np.linalg.norm(right_arm_vel - left_arm_vel)

    # 综合对称性奖励
    symmetry_reward = np.exp(-0.1 * pos_symmetry_error) * np.exp(-0.1 * vel_symmetry_error)

    return symmetry_reward


def _calc_arm_naturalness_reward(self):
    """
    计算手臂运动自然度奖励
    鼓励平滑、自然的手臂运动
    """
    # 获取手臂关节的加速度（通过速度变化估算）
    current_arm_vel = np.array(self._client.get_act_joint_velocities()[12:26])

    # 如果没有历史速度数据，初始化
    if not hasattr(self, 'prev_arm_vel'):
        self.prev_arm_vel = current_arm_vel.copy()
        return 0.5  # 初始奖励

    # 计算加速度
    arm_acceleration = np.abs(current_arm_vel - self.prev_arm_vel)

    # 保存当前速度供下一次使用
    self.prev_arm_vel = current_arm_vel.copy()

    # 鼓励低加速度（平滑运动）
    avg_acceleration = np.mean(arm_acceleration)
    naturalness_reward = np.exp(-0.5 * avg_acceleration)

    return naturalness_reward

#=======================手臂部分结束===================================================



def _calc_orient_reward(self, body_name):
    """身体朝向奖励"""
    body_quat = self._client.get_object_xquat_by_name(body_name, "OBJ_BODY")
    target_quat = np.array([1, 0, 0, 0])  # 中性朝向
    error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
    return np.exp(-error)


def _calc_body_orient_reward(self, body_name):
    """特定身体部位朝向奖励"""
    return self._calc_orient_reward(body_name)


##############################
##############################
# Define reward functions here
##############################
##############################

def _calc_fwd_vel_reward(self):
    """前进速度奖励：鼓励机器人以目标速度前进"""
    # root_vel = self._client.get_qvel()[0]  # 替代方法：从广义速度获取
    root_vel = self._client.get_body_vel(self._root_body_name, frame=1)[0][0]  # 获取根身体在世界坐标系下的X方向速度
    error = np.linalg.norm(root_vel - self._goal_speed_ref)  # 计算与目标速度的误差
    return np.exp(-10 * (error ** 2))  # 使用指数函数将误差映射到[0,1]区间


def _calc_yaw_vel_reward(self, yaw_vel_ref=0):
    """偏航角速度奖励：控制机器人的旋转速度"""
    yaw_vel = self._client.get_qvel()[5]  # 获取偏航角速度（绕Z轴旋转）
    error = np.linalg.norm(yaw_vel - yaw_vel_ref)  # 计算与目标偏航角速度的误差
    return np.exp(-10 * (error ** 3))  # 使用三次方惩罚大误差


def _calc_action_reward(self, prev_action):
    """动作平滑性奖励：鼓励连续动作之间的平滑变化"""
    action = self._client.get_pd_target()[0]  # 获取当前PD控制目标
    penalty = 5 * sum(np.abs(prev_action - action)) / len(action)  # 计算动作变化的平均绝对值
    return np.exp(-penalty)  # 惩罚大的动作变化


def _calc_torque_reward(self, prev_torque):
    """扭矩平滑性奖励：鼓励扭矩输出的平滑变化"""
    torque = np.asarray(self._client.get_act_joint_torques())  # 获取当前关节扭矩
    penalty = 0.25 * (sum(np.abs(prev_torque - torque)) / len(torque))  # 计算扭矩变化的平均绝对值
    return np.exp(-penalty)  # 惩罚大的扭矩变化


def _calc_height_reward(self):
    """高度奖励：维持合适的身体高度"""
    # 计算接触点高度（地面高度）
    if self._client.check_rfoot_floor_collision() or self._client.check_lfoot_floor_collision():
        contact_point = min([c.pos[2] for _, c in (self._client.get_rfoot_floor_contacts() +
                                                   self._client.get_lfoot_floor_contacts())])
    else:
        contact_point = 0

    current_height = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[2]  # 获取当前根身体高度
    relative_height = current_height - contact_point  # 计算相对于地面的高度
    error = np.abs(relative_height - self._goal_height_ref)  # 计算与目标高度的误差

    # 设置死区：在目标速度较高时允许更大的高度误差
    deadzone_size = 0.01 + 0.05 * self._goal_speed_ref
    if error < deadzone_size:
        error = 0  # 在小误差范围内不惩罚

    return np.exp(-40 * np.square(error))  # 高度误差的二次惩罚


def _calc_heading_reward(self):
    """朝向奖励：鼓励保持前进方向"""
    cur_heading = self._client.get_qvel()[:3]  # 获取线速度向量
    cur_heading /= np.linalg.norm(cur_heading)  # 归一化
    error = np.linalg.norm(cur_heading - np.array([1, 0, 0]))  # 计算与理想前进方向[1,0,0]的误差
    return np.exp(-error)  # 指数衰减奖励


def _calc_root_accel_reward(self):
    """根身体加速度奖励：鼓励平稳的运动"""
    qvel = self._client.get_qvel()  # 广义速度
    qacc = self._client.get_qacc()  # 广义加速度
    # 计算角速度和线加速度的惩罚
    error = 0.25 * (np.abs(qvel[3:6]).sum() + np.abs(qacc[0:3]).sum())
    return np.exp(-error)  # 惩罚大的加速度


def _calc_feet_separation_reward(self):
    """脚部分离奖励：维持合适的脚步间距"""
    rfoot_pos = self._client.get_rfoot_body_pos()[1]  # 右脚Y坐标
    lfoot_pos = self._client.get_lfoot_body_pos()[1]  # 左脚Y坐标
    foot_dist = np.abs(rfoot_pos - lfoot_pos)  # 计算双脚间距
    error = 5 * np.square(foot_dist - 0.35)  # 以0.35米为理想间距

    # 设置可接受的范围（0.30-0.40米）
    if foot_dist < 0.40 and foot_dist > 0.30:
        error = 0  # 在可接受范围内不惩罚

    return np.exp(-error)


def _calc_foot_frc_clock_reward(self, left_frc_fn, right_frc_fn):
    """基于时钟的脚部接触力奖励：根据步态相位协调脚部受力"""
    desired_max_foot_frc = self._client.get_robot_mass() * 9.8 * 0.5  # 期望最大脚力（一半体重）
    # desired_max_foot_frc = self._client.get_robot_mass()*10*1.2  # 替代计算方法

    # 归一化脚部受力到[-1,1]区间
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_frc *= 2
    normed_left_frc -= 1
    normed_right_frc *= 2
    normed_right_frc -= 1

    # 获取当前相位的时钟值
    left_frc_clock = left_frc_fn(self._phase)
    right_frc_clock = right_frc_fn(self._phase)

    # 使用tan函数计算得分（在期望接触时奖励接触，在期望摆动时惩罚接触）
    left_frc_score = np.tan(np.pi / 4 * left_frc_clock * normed_left_frc)
    right_frc_score = np.tan(np.pi / 4 * right_frc_clock * normed_right_frc)

    foot_frc_score = (left_frc_score + right_frc_score) / 2  # 平均双脚得分
    return foot_frc_score


def _calc_foot_vel_clock_reward(self, left_vel_fn, right_vel_fn):
    """基于时钟的脚部速度奖励：根据步态相位协调脚部运动"""
    desired_max_foot_vel = 0.2  # 期望最大脚部速度
    # 归一化脚部速度到[-1,1]区间
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_left_vel *= 2
    normed_left_vel -= 1
    normed_right_vel *= 2
    normed_right_vel -= 1

    # 获取当前相位的时钟值
    left_vel_clock = left_vel_fn(self._phase)
    right_vel_clock = right_vel_fn(self._phase)

    # 使用tan函数计算得分（在期望摆动时奖励运动，在期望支撑时惩罚运动）
    left_vel_score = np.tan(np.pi / 4 * left_vel_clock * normed_left_vel)
    right_vel_score = np.tan(np.pi / 4 * right_vel_clock * normed_right_vel)

    foot_vel_score = (left_vel_score + right_vel_score) / 2  # 平均双脚得分
    return foot_vel_score


def _calc_foot_pos_clock_reward(self):
    """基于时钟的脚部位置奖励：根据步态相位协调脚部高度"""
    desired_max_foot_height = 0.05  # 期望最大脚部抬升高度
    l_foot_pos = self._client.get_object_xpos_by_name('lf_force', 'OBJ_SITE')[2]  # 左脚高度
    r_foot_pos = self._client.get_object_xpos_by_name('rf_force', 'OBJ_SITE')[2]  # 右脚高度
    # 归一化脚部高度
    normed_left_pos = min(np.linalg.norm(l_foot_pos), desired_max_foot_height) / desired_max_foot_height
    normed_right_pos = min(np.linalg.norm(r_foot_pos), desired_max_foot_height) / desired_max_foot_height

    # 获取当前相位的时钟值
    left_pos_clock = self.left_clock[1](self._phase)
    right_pos_clock = self.right_clock[1](self._phase)

    # 使用tan函数计算得分
    left_pos_score = np.tan(np.pi / 4 * left_pos_clock * normed_left_pos)
    right_pos_score = np.tan(np.pi / 4 * right_pos_clock * normed_right_pos)

    foot_pos_score = left_pos_score + right_pos_score
    return foot_pos_score


def _calc_body_orient_reward(self, body_name, quat_ref=[1, 0, 0, 0]):
    """身体朝向奖励：维持身体特定部位的期望朝向"""
    body_quat = self._client.get_object_xquat_by_name(body_name, "OBJ_BODY")  # 获取身体四元数
    target_quat = np.array(quat_ref)  # 目标四元数
    # 计算四元数之间的误差（基于点积）
    error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
    return np.exp(-error)


def _calc_joint_vel_reward(self, enabled, cutoff=0.5):
    """关节速度奖励：限制关节速度在合理范围内"""
    motor_speeds = self._client.get_motor_velocities()  # 获取电机速度
    motor_limits = self._client.get_motor_speed_limits()  # 获取电机速度限制
    # 只考虑启用的关节
    motor_speeds = [motor_speeds[i] for i in enabled]
    motor_limits = [motor_limits[i] for i in enabled]
    # 计算超过限制阈值的关节的速度惩罚
    error = 5e-6 * sum([np.square(q)
                        for q, qmax in zip(motor_speeds, motor_limits)
                        if np.abs(q) > np.abs(cutoff * qmax)])
    return np.exp(-error)


def _calc_joint_acc_reward(self):
    """关节加速度奖励：限制关节加速度"""
    joint_acc_cost = np.sum(np.square(self._client.get_qacc()[-self._num_joints:]))  # 计算关节加速度的平方和
    return self.wp.joint_acc_weight * joint_acc_cost  # 乘以权重


def _calc_ang_vel_reward(self):
    """角速度奖励：限制身体角速度"""
    ang_vel = self._client.get_qvel()[3:6]  # 获取角速度向量
    ang_vel_cost = np.square(np.linalg.norm(ang_vel))  # 计算角速度大小的平方
    return self.wp.ang_vel_weight * ang_vel_cost  # 乘以权重


def _calc_impact_reward(self):
    """冲击奖励：减少脚部与地面的冲击力"""
    ncon = len(self._client.get_rfoot_floor_contacts()) + \
           len(self._client.get_lfoot_floor_contactts())  # 计算接触点数量
    if ncon == 0:
        return 0
    # 计算外部力的平方和，除以接触点数得到平均冲击
    quad_impact_cost = np.sum(np.square(self._client.get_body_ext_force())) / ncon
    return self.wp.impact_weight * quad_impact_cost  # 乘以权重


def _calc_zmp_reward(self):
    """零力矩点(ZMP)奖励：维持稳定性"""
    self.current_zmp = estimate_zmp(self)  # 估计当前ZMP
    # 防止ZMP估计的剧烈变化
    if np.linalg.norm(self.current_zmp - self._prev_zmp) > 1:
        self.current_zmp = self._prev_zmp
    zmp_cost = np.square(np.linalg.norm(self.current_zmp - self.desired_zmp))  # 计算与期望ZMP的误差
    self._prev_zmp = self.current_zmp  # 更新前一个ZMP
    return self.wp.zmp_weight * zmp_cost  # 乘以权重


def _calc_foot_contact_reward(self):
    """脚部接触奖励：限制脚部接触点的位置"""
    right_contacts = self._client.get_rfoot_floor_collisions()  # 右脚接触点
    left_contacts = self._client.get_lfoot_floor_collisions()  # 左脚接触点

    radius_thresh = 0.3  # 距离阈值
    f_base = self._client.get_qpos()[0:2]  # 基座位置
    # 计算超过阈值的接触点距离
    c_dist_r = [(np.linalg.norm(c.pos[0:2] - f_base)) for _, c in right_contacts]
    c_dist_l = [(np.linalg.norm(c.pos[0:2] - f_base)) for _, c in left_contacts]
    d = sum([r for r in c_dist_r if r > radius_thresh] +
            [r for r in c_dist_l if r > radius_thresh])
    return self.wp.foot_contact_weight * d  # 乘以权重


def _calc_gait_reward(self):
    """步态奖励：基于相位的步态协调奖励"""
    if self._period <= 0:
        raise Exception("Cycle period should be greater than zero.")

    # 获取脚部地面反作用力
    rfoot_grf = self._client.get_rfoot_grf()
    lfoot_grf = self._client.get_lfoot_grf()

    # 获取脚部速度
    rfoot_speed = self._client.get_rfoot_body_speed()
    lfoot_speed = self._client.get_lfoot_body_speed()

    # 获取脚部位置
    rfoot_pos = self._client.get_rfoot_body_pos()
    lfoot_pos = self._client.get_lfoot_body_pos()
    swing_height = 0.3  # 摆动相期望高度
    stance_height = 0.1  # 支撑相期望高度

    r = 0.5  # 相位分割点
    if self._phase < r:
        # 右脚支撑期，左脚摆动期
        cost = (0.01 * lfoot_grf)  # 主要惩罚摆动脚的接触力
        # + np.square(lfoot_pos[2]-swing_height)   # 可选的脚部高度惩罚
        # + (10*np.square(rfoot_pos[2]-stance_height))  # 可选的支撑脚高度惩罚
    else:
        # 左脚支撑期，右脚摆动期
        cost = (0.01 * rfoot_grf)  # 主要惩罚摆动脚的接触力
        # + np.square(rfoot_pos[2]-swing_height)
        # + (10*np.square(lfoot_pos[2]-stance_height))
    return self.wp.gait_weight * cost  # 乘以权重


def _calc_reference(self):
    """参考轨迹奖励：跟踪预定义的参考轨迹"""
    if self.ref_poses is None:
        raise Exception("Reference trajectory not provided.")

    # 根据相位获取参考姿态
    phase = self._phase
    traj_length = self.traj_len
    indx = int(phase * (traj_length - 1))
    reference_pose = self.ref_poses[indx, :]

    # 获取当前姿态
    current_pose = np.array(self._client.get_act_joint_positions())

    # 计算与参考姿态的误差
    cost = np.square(np.linalg.norm(reference_pose - current_pose))
    return self.wp.ref_traj_weight * cost  # 乘以权重


##############################
##############################
# Define utility functions
##############################
##############################

def estimate_zmp(self):
    """估计零力矩点(ZMP) - 动态稳定性的重要指标"""
    Gv = 9.80665  # 重力加速度
    Mg = self._mass * Gv  # 机器人重量

    # 获取质心位置、线动量和角动量
    com_pos = self._sim.data.subtree_com[1].copy()
    lin_mom = self._sim.data.subtree_linvel[1].copy() * self._mass
    ang_mom = self._sim.data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)

    # 计算动量的变化率
    d_lin_mom = (lin_mom - self._prev_lin_mom) / self._control_dt
    d_ang_mom = (ang_mom - self._prev_ang_mom) / self._control_dt

    Fgz = d_lin_mom[2] + Mg  # Z方向的总力

    # 检查与地面的接触
    contacts = [self._sim.data.contact[i] for i in range(self._sim.data.ncon)]
    contact_flag = [(c.geom1 == 0 or c.geom2 == 0) for c in contacts]

    # 计算ZMP位置
    if (True in contact_flag) and Fgz > 20:  # 如果有接触且Z方向力足够大
        zmp_x = (Mg * com_pos[0] - d_ang_mom[1]) / Fgz
        zmp_y = (Mg * com_pos[1] + d_ang_mom[0]) / Fgz
    else:
        zmp_x = com_pos[0]  # 无接触时ZMP在质心投影
        zmp_y = com_pos[1]

    # 保存当前动量用于下一帧计算
    self._prev_lin_mom = lin_mom
    self._prev_ang_mom = ang_mom
    return np.array([zmp_x, zmp_y])


##############################
##############################
# Based on apex
##############################
##############################

def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, FREQ=40):
    """创建基于相位的奖励函数 - 核心的步态协调机制"""

    from scipy.interpolate import PchipInterpolator  # 用于创建平滑的插值函数

    # 将时间转换为相位长度（控制步数）
    right_swing = np.array([0.0, swing_duration]) * FREQ  # 右脚摆动期
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ  # 第一次双脚支撑期
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ  # 左脚摆动期
    second_dblstance = np.array(
        [2 * swing_duration + stance_duration, 2 * (swing_duration + stance_duration)]) * FREQ  # 第二次双脚支撑期

    # 初始化相位点数组
    r_frc_phase_points = np.zeros((2, 8))  # [相位位置, 奖励值]
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    # 设置右脚摆动期的相位点
    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 0] = r_frc_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0, 1] = r_frc_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0, 0] = r_vel_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0, 1] = r_vel_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset

    # 在右脚摆动期：期望左脚接触，右脚运动
    l_vel_phase_points[1, :2] = r_frc_phase_points[1, :2] = np.negative(np.ones(2))  # 惩罚左脚速度和右脚接触力
    l_frc_phase_points[1, :2] = r_vel_phase_points[1, :2] = np.ones(2)  # 奖励左脚接触力和右脚速度

    # 设置第一次双脚支撑期的相位点
    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 2] = r_frc_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 3] = r_frc_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 2] = r_vel_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 3] = r_vel_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset

    # 根据支撑模式设置奖励值
    if stance_mode == "aerial":
        # 空中期：期望脚部运动，不期望接触
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.negative(np.ones(2))  # 惩罚双脚接触力
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.ones(2)  # 奖励双脚速度
    elif stance_mode == "zero":
        # 零期望：不奖励也不惩罚
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.zeros(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.zeros(2)
    else:
        # 地面行走：期望脚部接触，不期望运动
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.ones(2)  # 奖励双脚接触力
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.negative(np.ones(2))  # 惩罚双脚速度

    # 设置左脚摆动期的相位点
    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 4] = r_frc_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0, 5] = r_frc_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0, 4] = r_vel_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0, 5] = r_vel_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset

    # 在左脚摆动期：期望右脚接触，左脚运动
    l_vel_phase_points[1, 4:6] = r_frc_phase_points[1, 4:6] = np.ones(2)  # 奖励左脚速度和右脚接触力
    l_frc_phase_points[1, 4:6] = r_vel_phase_points[1, 4:6] = np.negative(np.ones(2))  # 惩罚左脚接触力和右脚速度

    # 设置第二次双脚支撑期的相位点
    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 6] = r_frc_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 7] = r_frc_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 6] = r_vel_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 7] = r_vel_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset

    # 根据支撑模式设置奖励值（与第一次双脚支撑期相同）
    if stance_mode == "aerial":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.negative(np.ones(2))
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.ones(2)
    elif stance_mode == "zero":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.zeros(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.zeros(2)
    else:
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.ones(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.negative(np.ones(2))

    ## 将数据扩展到三个周期：确保连续性
    # 前一个周期
    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = r_frc_phase_points[0] - r_frc_phase_points[
        0, -1] - dbl_stance_relax_offset
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = r_vel_phase_points[0] - r_vel_phase_points[
        0, -1] - dbl_stance_relax_offset

    # 后一个周期
    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = r_frc_phase_points[0] + r_frc_phase_points[
        0, -1] + dbl_stance_relax_offset
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = r_vel_phase_points[0] + r_vel_phase_points[
        0, -1] + dbl_stance_relax_offset

    # 合并三个周期的数据
    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    ## 使用PCHIP插值创建平滑的时钟函数
    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])

    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline]