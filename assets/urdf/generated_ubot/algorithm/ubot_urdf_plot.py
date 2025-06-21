import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

#modification on contact distance
sim_params.physx.contact_offset=0.002
sim_params.physx.rest_offset=0.0001

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) 
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 4
spacing = 1
env_lower = gymapi.Vec3(-0,-0, -0 )
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# add cartpole urdf asset

# asset_root = "../structure_library/library/1/"
asset_root = "../"

# asset_file = "mjcf/exoskeleton/model/std.xml"
#asset_file = "urdf/shell/shell.urdf"
#asset_root = "../../assets"
asset_file = "ubot.urdf"
#asset_file = "urdf/PAI-urdf/urdf/PAI-urdf.urdf"
#asset_file = "urdf/LegBot/urdf/LegBot.urdf"
# asset_root = "../../assets"
# asset_file = "mjcf/nv_ant.xml"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
asset_options.vhacd_enabled = True
asset_options.vhacd_params.resolution = 500000
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
ubot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
#initial_pose.r = gymapi.Quat(-0.707, 0.0, 0.0, 0.707)
initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# Create environment
env = gym.create_env(sim, env_lower, env_upper, 2)
ubot = gym.create_actor(env, ubot_asset, initial_pose, 'ubot', 0, 0)
# Configure DOF properties
props = gym.get_actor_dof_properties(env, ubot)
print(gymapi.DOF_MODE_POS)
props['lower'].fill(-math.pi/2)
props['upper'].fill(math.pi/2)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(5000.0)
props["damping"].fill(100.0)


gym.set_actor_dof_properties(env, ubot, props)
# Set DOF drive targets
# r_dof_handle = gym.find_actor_dof_handle(env, ubot, 'LJ')
# l_dof_handle = gym.find_actor_dof_handle(env, ubot, 'RJ')
# gym.set_dof_target_position(env, r_dof_handle, 0)
# gym.set_dof_target_position(env, l_dof_handle, 0)
# gym.set_dof_target_velocity(env, r_dof_handle, 1.0)
# gym.set_dof_target_position(env, l_dof_handle, 0.0)
# Look at the first env
cam_pos = gymapi.Vec3(2,0,1)
cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# actor_root_state = gym.acquire_actor_root_state_tensor(sim)
# root_states = gymtorch.wrap_tensor(actor_root_state)

# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    # Wait for dt to elapse in real time
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    # gym.refresh_actor_root_state_tensor(sim)
    # print(root_states[3:7])

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
