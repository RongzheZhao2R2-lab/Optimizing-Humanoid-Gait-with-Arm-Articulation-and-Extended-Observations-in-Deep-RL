import sys
import os
from dm_control import mjcf
import random
import string

JVRC_DESCRIPTION_PATH = "models/jvrc_mj_description/xml/scene.xml"


def builder(export_path):
    """构建并修改JVRC机器人的MuJoCo模型，然后导出到指定路径"""

    print("Modifying XML model for arm control...")
    # 从原始XML文件加载MJCF模型
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)

    # 设置MuJoCo的仿真参数
    # njmax: 最大约束数，增加以提高仿真稳定性
    mjcf_model.size.njmax = 1200
    # nconmax: 最大接触数，增加以处理更多接触情况
    mjcf_model.size.nconmax = 400

    # 移除所有原有的碰撞设置，重新定义
    mjcf_model.contact.remove()

    # 定义各种关节分类
    waist_joints = ['WAIST_Y', 'WAIST_P', 'WAIST_R']  # 腰部关节
    head_joints = ['NECK_Y', 'NECK_R', 'NECK_P']  # 头部关节
    hand_joints = ['R_UTHUMB', 'R_LTHUMB', 'R_UINDEX', 'R_LINDEX', 'R_ULITTLE', 'R_LLITTLE',
                   'L_UTHUMB', 'L_LTHUMB', 'L_UINDEX', 'L_LINDEX', 'L_ULITTLE', 'L_LLITTLE']  # 手部关节（手指）

    # 修改：保留手臂关节，用于手臂控制
    arm_joints = ['R_SHOULDER_Y', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y',
                  'L_SHOULDER_Y', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y']  # 手臂关节（部分）

    # 修改：添加更多手臂关节以支持完整的手臂控制
    full_arm_joints = [
        'R_SHOULDER_P', 'R_SHOULDER_R', 'R_SHOULDER_Y',  # 右肩全部3个自由度
        'R_ELBOW_P', 'R_ELBOW_Y',  # 右肘2个自由度
        'R_WRIST_R', 'R_WRIST_Y',  # 右腕2个自由度
        'L_SHOULDER_P', 'L_SHOULDER_R', 'L_SHOULDER_Y',  # 左肩全部3个自由度
        'L_ELBOW_P', 'L_ELBOW_Y',  # 左肘2个自由度
        'L_WRIST_R', 'L_WRIST_Y'  # 左腕2个自由度
    ]

    leg_joints = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
                  'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P']  # 腿部关节

    # 修改：保留腿部和手臂的执行器，只移除腰部、头部和手部的执行器
    for mot in mjcf_model.actuator.motor:
        # 如果关节不在腿部和手臂关节列表中，则移除执行器
        if mot.joint.name not in leg_joints + full_arm_joints:
            mot.remove()

    # 修改：只移除未使用的关节（腰部、头部、手部），保留手臂关节
    for joint in waist_joints + head_joints + hand_joints:
        try:
            mjcf_model.find('joint', joint).remove()
        except:
            print(f"警告: 未找到关节 {joint}，跳过移除")

    # 移除原有的等式约束（如手指的联动约束）
    mjcf_model.equality.remove()

    # 修改：注释掉手臂关节的等式约束，因为我们希望主动控制手臂
    # 不再将手臂关节固定到特定位置
    '''
    # 为手臂关节添加新的等式约束（注释掉，因为我们希望主动控制手臂）
    arm_joints = ['R_SHOULDER_P', 'R_SHOULDER_R', 'R_ELBOW_P',
                  'L_SHOULDER_P', 'L_SHOULDER_R', 'L_ELBOW_P']

    # 添加关节等式约束，polycoef参数定义关节位置的多项式系数
    # 这里使用常数系数将关节固定在特定角度
    mjcf_model.equality.add('joint', joint1=arm_joints[0], polycoef='-0.052 0 0 0 0')  # 右肩俯仰
    mjcf_model.equality.add('joint', joint1=arm_joints[1], polycoef='-0.169 0 0 0 0')  # 右肩横滚
    mjcf_model.equality.add('joint', joint1=arm_joints[2], polycoef='-0.523 0 0 0 0')  # 右肘俯仰
    mjcf_model.equality.add('joint', joint1=arm_joints[3], polycoef='-0.052 0 0 0 0')  # 左肩俯仰
    mjcf_model.equality.add('joint', joint1=arm_joints[4], polycoef='0.169 0 0 0 0')   # 左肩横滚
    mjcf_model.equality.add('joint', joint1=arm_joints[5], polycoef='-0.523 0 0 0 0')  # 左肘俯仰
    '''

    # 修改：扩展碰撞几何体列表，包含手臂相关部位
    # 定义保留碰撞检测的几何体列表
    # 现在包含腿部和手臂的主要部位
    collision_geoms = [
        # 腿部
        'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',
        'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',
        # 手臂（新增）
        'R_SHOULDER_P_S', 'R_SHOULDER_R_S', 'R_SHOULDER_Y_S', 'R_ELBOW_P_S', 'R_ELBOW_Y_S',
        'L_SHOULDER_P_S', 'L_SHOULDER_R_S', 'L_SHOULDER_Y_S', 'L_ELBOW_P_S', 'L_ELBOW_Y_S'
    ]

    # 移除未使用的碰撞几何体
    for body in mjcf_model.worldbody.find_all('body'):
        for idx, geom in enumerate(body.geom):
            # 为几何体重新命名，便于识别
            geom.name = body.name + '-geom-' + repr(idx)
            # 如果几何体属于碰撞类，且所在身体不在保留列表中，则移除
            if (geom.dclass.dclass == "collision"):
                if body.name not in collision_geoms:
                    geom.remove()

    # 手动为脚部创建碰撞几何体
    # 使用简单的长方体代替复杂的网格碰撞体，提高仿真效率
    mjcf_model.worldbody.find('body', 'R_ANKLE_P_S').add('geom',
                                                         dclass='collision',
                                                         size='0.1 0.05 0.01',  # 长方体尺寸：长、宽、高
                                                         pos='0.029 0 -0.09778',  # 位置偏移
                                                         type='box'  # 几何体类型：长方体
                                                         )
    mjcf_model.worldbody.find('body', 'L_ANKLE_P_S').add('geom',
                                                         dclass='collision',
                                                         size='0.1 0.05 0.01',
                                                         pos='0.029 0 -0.09778',
                                                         type='box'
                                                         )

    # 修改：添加手臂碰撞几何体（可选）
    # 为手臂添加简单的碰撞几何体，提高仿真稳定性
    try:
        # 右前臂碰撞体
        mjcf_model.worldbody.find('body', 'R_ELBOW_P_S').add('geom',
                                                             dclass='collision',
                                                             size='0.02 0.02 0.1',  # 圆柱形近似
                                                             pos='0 0 -0.1',
                                                             type='cylinder'
                                                             )
        # 左前臂碰撞体
        mjcf_model.worldbody.find('body', 'L_ELBOW_P_S').add('geom',
                                                             dclass='collision',
                                                             size='0.02 0.02 0.1',
                                                             pos='0 0 -0.1',
                                                             type='cylinder'
                                                             )
    except:
        print("警告: 无法添加手臂碰撞几何体")

    # 添加碰撞排除规则，避免不必要的自碰撞检测
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')  # 右膝和右脚踝
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')  # 左膝和左脚踝

    # 修改：添加手臂相关的碰撞排除规则
    mjcf_model.contact.add('exclude', body1='R_SHOULDER_P_S', body2='R_SHOULDER_Y_S')  # 右肩关节间
    mjcf_model.contact.add('exclude', body1='L_SHOULDER_P_S', body2='L_SHOULDER_Y_S')  # 左肩关节间
    mjcf_model.contact.add('exclude', body1='R_ELBOW_Y_S', body2='R_WRIST_Y_S')  # 右肘和右腕
    mjcf_model.contact.add('exclude', body1='L_ELBOW_Y_S', body2='L_WRIST_Y_S')  # 左肘和左腕

    # 移除未使用的网格资源，减小文件大小
    # 收集所有仍在使用的网格名称
    meshes = [g.mesh.name for g in mjcf_model.find_all('geom') if g.type == 'mesh' or g.type == None]
    # 移除未被任何几何体使用的网格
    for mesh in mjcf_model.find_all('mesh'):
        if mesh.name not in meshes:
            mesh.remove()

    # 调整力传感器站点的位置
    # 将这些站点移动到脚部几何体的合适位置
    mjcf_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'  # 右脚力传感器
    mjcf_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'  # 左脚力传感器

    # 修改：调整手部力传感器位置（如果存在）
    try:
        mjcf_model.worldbody.find('site', 'rh_force').pos = '0.0 0.0 -0.05'  # 右手力传感器
        mjcf_model.worldbody.find('site', 'lh_force').pos = '0.0 0.0 -0.05'  # 左手力传感器
    except:
        print("警告: 未找到手部力传感器站点")

    # 添加多个长方体几何体，可能用于创建地面或障碍物
    for idx in range(20):
        mjcf_model.worldbody.add('geom',
                                 name='box' + repr(idx + 1).zfill(2),  # 名称格式：box01, box02, ...
                                 pos='0 0 -0.2',  # 位置
                                 dclass='collision',  # 使用碰撞类
                                 group='0',  # 组别0（碰撞组）
                                 size='1 1 0.1',  # 尺寸：大而薄的长方体
                                 type='box',  # 几何体类型
                                 material=''  # 无特定材质
                                 )

    # 导出修改后的模型
    # 包括所有资源文件（网格、纹理等）
    mjcf.export_with_assets(mjcf_model,
                            out_dir=os.path.dirname(export_path),  # 输出目录
                            out_file_name=export_path,  # 输出文件名
                            precision=5  # 数值精度
                            )
    print("Exporting XML model with arm control to ", export_path)
    return


if __name__ == '__main__':
    # 当脚本直接运行时，从命令行参数获取导出路径
    builder(sys.argv[1])