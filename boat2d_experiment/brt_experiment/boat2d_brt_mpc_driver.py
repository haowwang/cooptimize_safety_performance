import casadi as ca
import plotly.graph_objects as go
import numpy as np 
from boat2d_brt_mpc_utils import *
import time 
from scipy.io import loadmat, savemat

initial_states_file = "experiments_data/boat2d_brt_oc_rollout_test_initial_states.mat"
rollout_points = loadmat(initial_states_file)['initial_states']
rollout_points = np.array(rollout_points).T

goalX = 1.5
goalY = 0
goalR = 0.25

obs1_x = -0.5
obs1_y = 0.5
obs1_w = 0.8
obs1_h = 0.8

obs2_x = -1
obs2_y = -1.25
obs2_w = 0.4
obs2_h = 1.5

boundary_wall = [-3, 2, -2, 2] # [x_min, x_max, y_min, y_max]

dt = 0.005
time_horizon = 2
total_time_steps = int(time_horizon / dt)
planning_horizon = 70

def dynamics(x, u):
    dx = [u[0] + 2 - 0.5 * x[1] ** 2, u[1]]
    return dx

# x0 = [-1, 0]
rollout_costs = np.zeros(rollout_points.shape[0])
rollout_times = np.zeros(rollout_points.shape[0])
env = None
state_trajectories = np.zeros((rollout_points.shape[0], total_time_steps + 1, 2))
ctrl_trajectories = np.zeros((rollout_points.shape[0], total_time_steps, 2))
for i in range(rollout_points.shape[0]):
    print(f"Rollout {i+1} of {rollout_points.shape[0]}")
    x0 = rollout_points[i, :].reshape(1, 2)[0]
    rollout_costs[i], state_trajectories[i, :, :], ctrl_trajectories[i, :, :] \
    ,rollout_times[i], env = run_mpc_rollout(x0, [goalX, goalY, goalR], 
        [[obs1_x, obs1_y, obs1_w, obs1_h],
        [obs2_x, obs2_y, obs2_w, obs2_h]], boundary_wall, 
        planning_horizon, dt, total_time_steps, dynamics, env)

print(rollout_times)
env.write_image('mpc_rollout.png')
savemat('experiments_data/boat2d_mpc_rollout_data.mat',
                 {'rollout_costs': rollout_costs,
                 'rollout_trajectories': state_trajectories,
                 'ctrl_trajectories': ctrl_trajectories,
                 'rollout_times': rollout_times})