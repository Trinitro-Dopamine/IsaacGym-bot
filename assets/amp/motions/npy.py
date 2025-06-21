import numpy as np
import csv
from scipy.spatial.transform import Rotation


def quaternion_to_euler(quaternion):
    r = Rotation.from_quat(quaternion)
    euler = r.as_euler('XYZ', degrees=True) / 180 * np.pi  # 内旋 XYZ
    return euler


data = np.load('./amp_humanoid_walk.npy', allow_pickle=True).item()
num = data['rotation']['arr'].shape[0]
print(num)
print(data.keys())
print(data['rotation']['arr'].shape) #[154,15,4]
print(data['rotation'])
euler = np.zeros((num, 15, 3))
for i in range(data['rotation']['arr'].shape[0]):
    for n in range(data['rotation']['arr'].shape[1]):
        euler[i][n] = quaternion_to_euler(data['rotation']['arr'][i][n])
# print(euler)

rotation = euler.reshape(num, 45)
root_translation = data['root_translation']['arr']
global_velocity = data['global_velocity']['arr'].reshape(num, 45)
global_angular_velocity = data['global_angular_velocity']['arr'].reshape(num, 45)

header = ['pelvis', '', '', 'torso', '', '', 'head', '', '', 'right_upper_arm', '', '', 'right_lower_arm', '', '',
          'right_hand', '', '', 'left_upper_arm', '', '', 'left_lower_arm', '', '', 'left_hand', '', '', 'right_thigh',
          '', '', 'right_shin', '', '', 'right_foot', '', '', 'left_thigh', '', '', 'left_shin', '', '', 'left_foot', '', '']

# with open('./data/backflip_rotation.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(header)
#     csv_writer.writerows(rotation)

# with open('./data/backflip_root_translation.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     # csv_writer.writerow(header)
#     csv_writer.writerows(root_translation)

# with open('./data/backflip_global_velocity.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(header)
#     csv_writer.writerows(global_velocity)

# with open('./data/backflip_global_angular_velocity.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(header)
#     csv_writer.writerows(global_angular_velocity)
