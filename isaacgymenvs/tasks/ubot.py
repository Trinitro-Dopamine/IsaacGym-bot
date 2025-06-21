# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#python train.py task=Ubot wandb_activate=True sim_device=cpu
#python train.py task=Ubot sim_device=cpu checkpoint=runs/Ubot/nn/runs/Ubot/nn/last_Ubot_ep_850_rew_11.2767725.pth test=True num_envs=10

#TODO: stop convexing my hull and start using collision(by deleting the core mesh)
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from isaacgym.terrain_utils import *
from typing import Tuple, Dict

class bcolors:
    GREEN = '\033[92m' #GREEN
    YELLOW = '\033[93m' #YELLOW
    RED = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

class Ubot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["force"] = self.cfg["env"]["learn"]["forceRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        #termination space
        self.termination_height = self.cfg["env"]['termination']["terminationHeight"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state


        #TODO: automate fill numActions,numObservations
        self.cfg["env"]["numObservations"] = 18
        self.cfg["env"]["numActions"] = 2
        # print(self.num_dof)
        # garbage=[]
        # input(garbage)
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.num_bodies = self.rigid_body_states.shape[1]
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]


        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_dof):        
            self.default_dof_pos[:, i] = 0.0

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        terrain_type = self.cfg["env"]["plane"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = False
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        terrain_width = 120.
        terrain_length = 120.
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.002  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
        heightfield[:, :] = random_uniform_terrain(new_sub_terrain(), min_height=0.0, max_height=0.1, step=0.02, downsampled_scale=0.5).height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -2.
        tm_params.transform.p.y = -2.
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = 0.1
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/shell/ubot.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = False
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.004
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 500000

        ubot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ubot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ubot_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(ubot_asset)
        self.dof_names = self.gym.get_asset_dof_names(ubot_asset)

        # extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        # feet_names = [s for s in body_names if extremity_name in s]
        # self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        # knee_names = [s for s in body_names if "THIGH" in s]
        # self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)

        dof_props = self.gym.get_asset_dof_properties(ubot_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.ubot_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            ubot_handle = self.gym.create_actor(env_ptr, ubot_asset, start_pose, None, i, 0)
            self.gym.set_actor_dof_properties(env_ptr, ubot_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ubot_handle)
            self.envs.append(env_ptr)
            self.ubot_handles.append(ubot_handle)


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ubot_reward(
            # tensors
            self.root_states,#maybe I can get rid of this
            self.rigid_body_states,
            self.commands,
            self.torques,
            self.contact_forces,
            # self.knee_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.termination_height,
            self.max_episode_length,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf[:] = compute_ubot_observations(  # tensors
                                                        self.root_states,
                                                        self.rigid_body_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        #print(self.commands_x[env_ids],self.commands_y[env_ids],self.commands_yaw[env_ids])

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ubot_reward(
    # tensors
    root_states,
    rigid_body_states,
    commands,
    torques,
    contact_forces,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    terminationHeight,
    max_episode_length
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float],float, int) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_pos=root_states[:, 0:3]
    base_lin_vel = quat_rotate_inverse(base_quat,torch.mean(rigid_body_states[:,:,7:10],dim=1))
    #base_ang_vel = quat_rotate_inverse(base_quat, torch.mean(rigid_body_states[:,:,10:13],dim=1))
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1) 
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    contact_force=torch.norm(contact_forces[:,:,:],dim=2,p=2)

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]
    rew_force = torch.sum(torch.square(contact_force), dim=1)*rew_scales["force"]
    
    #height_error=torch.square(rigid_body_states[:,16, 2] - 0.136)
    #rew_height=torch.exp(-height_error/0.25)* 0.02
    # penalty_pos=torch.where(base_pos[:,1]<1.2)
    
    # # reset agents
    reset_force=(torch.norm(contact_force[:,:],dim=1,p=torch.inf)>500)
    reset_lin=lin_vel_error>1.5
    reset_time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs

    #reset_height=(rigid_body_states[:,16, 2]>1.5)|(rigid_body_states[:,16, 2]<0.1)
    
    reset=reset_force|reset_lin#|reset_height
    #rew_ang_reset=torch.where(reset!=0,-0.2,0.0)

    #print(rew_lin_vel_xy[0],rew_ang_vel_z[0],rew_torque[0],rew_force[0],rew_height[0])
    total_reward = rew_lin_vel_xy + rew_ang_vel_z #+ rew_torque + rew_force #+rew_height

    total_reward = torch.clip(total_reward, 0., None)
    return total_reward.detach(), reset|reset_time_out


@torch.jit.script
def compute_ubot_observations(root_states,
                                rigid_body_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    #related average coordinate


    base_lin_vel = quat_rotate_inverse(base_quat,torch.mean(rigid_body_states[:,:,7:10],dim=1))*lin_vel_scale
    #base_ang_vel = quat_rotate_inverse(base_quat, torch.mean(rigid_body_states[:,:,10:13],dim=1))*ang_vel_scale
    # base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale

    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs

