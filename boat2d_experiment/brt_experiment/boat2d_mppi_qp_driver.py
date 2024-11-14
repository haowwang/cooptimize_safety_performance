import torch
import time
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator
from boat2d_mppi_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()

num_rollouts = 500000

initial_states_file = "experiments_data/boat2d_brt_oc_rollout_test_initial_states.mat"

data_file = 'experiments_data/boat2d_brt_ctrl_cons_data_gurobi.mat'

rollout_points = loadmat(initial_states_file)['initial_states']
rollout_points = torch.tensor(rollout_points).T
rollout_points = rollout_points.to(device)

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
horizon = 2
total_time_steps = int(horizon / dt)
planning_horizon = 20 # for mppi_qp

softmax_lambda = 200.0
safety_threshold = 0.08

# dynamics have to be code into get_QP_ctrl also
def dynamics(x, u, device):
    dx = torch.zeros_like(x, device=device)
    dx[:, 0] = u[:, 0] + 2 - 0.5 * x[:, 1] ** 2
    dx[:, 1] = u[:, 1]
    return dx

matlab_var_dict = get_matlab_variables(data_file)
eval_deriv_v1_x, eval_deriv_v1_y, eval_deriv_v1_t \
            = get_deriv_v1_eval_func(matlab_var_dict)
eval_v1_data = get_v1_data_eval_func(matlab_var_dict)
eval_opt_ctrl_x, eval_opt_ctrl_y \
    = get_opt_ctrl_v1_eval_func(matlab_var_dict)


rollout_costs = torch.zeros(rollout_points.shape[0], device=device)
rollout_times = torch.zeros(rollout_points.shape[0], device=device)

env = None
state_trajectories = torch.zeros(rollout_points.shape[0], 
                        total_time_steps + 1, 2, device=device)
ctrl_trajectories = torch.zeros(rollout_points.shape[0], 
                        total_time_steps, 2, device=device)

for i in range(rollout_points.shape[0]):
    print(f"Rollout {i+1} of {rollout_points.shape[0]}")
    x0 = rollout_points[i, :].reshape(1, 2)
    rollout_cost, env, state_trajectories[i, :, :], \
        ctrl_trajectories[i, :, :], _, rollout_time \
                          = run_batch_mppi_qp(env, device, 
                            x0, dynamics, total_time_steps, num_rollouts,
                            planning_horizon, dt, [goalX, goalY, goalR],
                            [[obs1_x, obs1_y, obs1_w, obs1_h],
                            [obs2_x, obs2_y, obs2_w, obs2_h]],
                            boundary_wall, softmax_lambda,
                            eval_deriv_v1_x, eval_deriv_v1_y,
                            eval_deriv_v1_t, eval_v1_data,
                            eval_opt_ctrl_x, eval_opt_ctrl_y,
                            safety_threshold)
    rollout_costs[i] = rollout_cost.item()
    rollout_times[i] = rollout_time

print(rollout_times)
rollout_costs = rollout_costs.cpu()
rollout_times = rollout_times.cpu()
state_trajectories = state_trajectories.cpu()
ctrl_trajectories = ctrl_trajectories.cpu()
env.write_image('mppi_qp_rollout.png')
savemat('experiments_data/boat2d_mppi_qp_rollout_data.mat',
         {'rollout_costs': rollout_costs.numpy(),
           'rollout_trajectories': state_trajectories.numpy(),
           'ctrl_trajectories': ctrl_trajectories.numpy(),
           'rollout_times': rollout_times.numpy()})
print(f"Time taken is {time.time() - start_time:.2f} seconds")
