import torch
import time
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
import cvxpy as cp

def signed_distance_to_rectangle(points, rectangle, device):
    # points: (N, num_steps, 2)
    # rectangle: (cx, cy, width, height)
    # returns: (N, num_steps) signed distance to rectangle
    cx, cy = rectangle[0], rectangle[1]
    rect_width = rectangle[2]
    rect_height = rectangle[3]
    half_width = rect_width / 2.0
    half_height = rect_height / 2.0
    
    left = cx - half_width
    right = cx + half_width
    bottom = cy - half_height
    top = cy + half_height
    
    px = points[:, :, 0]
    py = points[:, :, 1]
    
    dx = torch.maximum(torch.maximum(left - px, px - right),
                        torch.tensor(0.0, device=device))
    dy = torch.maximum(torch.maximum(bottom - py, py - top),
                       torch.tensor(0.0, device=device))
    
    outside_distance = torch.sqrt(dx**2 + dy**2)
    inside_distance_x = torch.minimum(px - left, right - px)
    inside_distance_y = torch.minimum(py - bottom, top - py)
    inside_distance = torch.minimum(inside_distance_x, inside_distance_y)
    inside = (left <= px) & (px <= right) & (bottom <= py) & (py <= top)
    
    signed_distance = torch.where(inside, -inside_distance, outside_distance)
    return signed_distance

def mppi_cost_func(x, u, goal, obs, boundary_wall, device):
    # x: [N, num_steps, state_dim]
    # u: [N, num_steps, control_dim]
    # cost: [N]
    goalX, goalY, goalR = goal
    obs1_x, obs1_y, obs1_w, obs1_h = obs[0]
    obs2_x, obs2_y, obs2_w, obs2_h = obs[1]
    cost = torch.zeros(x.shape[0])
    x1 = x[:, :, 0]
    x2 = x[:, :, 1]
    goal_distance = torch.sqrt((x1 - goalX)**2 + (x2 - goalY)**2)
    obs1_distance = -signed_distance_to_rectangle(x,
                    torch.tensor([obs1_x, obs1_y, obs1_w, obs1_h]), device)
    obs2_distance = -signed_distance_to_rectangle(x,
                     torch.tensor([obs2_x, obs2_y, obs2_w, obs2_h]), device)
    obs_distance = torch.max(obs1_distance, obs2_distance)
    left_wall_distance = (-x1 + boundary_wall[0]) 
    right_wall_distance = (x1 - boundary_wall[1])
    bottom_wall_distance = (-x2 + boundary_wall[2])
    top_wall_distance = (x2 - boundary_wall[3])
    wall_distance = torch.max(torch.max(left_wall_distance, right_wall_distance),
                              torch.max(top_wall_distance, bottom_wall_distance))
    obstacle_cost = torch.max(torch.tensor(0.0), torch.max(obs_distance, wall_distance))
    cost = goal_distance + 1e12 * obstacle_cost
    cost = torch.sum(cost, dim=1)
    return cost

def dyn_propogate(x, u, dynamics, dt, device):
    # x: [N, state_dim]
    # u: [N, control_dim]
    # x_next: [N, state_dim]
    dx = dynamics(x, u, device)
    x_next = x + dx * dt
    return x_next

def batch_rollout_traj(x0, u, dynamics, dt, device):
    # x0: [N, state_dim]
    # u: [N, num_steps, control_dim]
    # x: [N, num_steps, state_dim]
    x = torch.zeros(u.shape[0], u.shape[1], x0.shape[1], device=device) # last arg was 2
    x[:, 0, :] = x0
    for i in range(1, u.shape[1]):
        x[:, i, :] = dyn_propogate(x[:, i-1, :], u[:, i-1, :], dynamics, dt, device)
    return x

def plot_traj(fig, x_dev, plot_color='blue'):
    x = x_dev
    x = x.cpu()
    if plot_color == 'red':
        fig.add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode='markers',
            name='Trajectory', fillcolor=plot_color, line=dict(width=2)))
    else :
        fig.add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode='lines',
             name='Trajectory', fillcolor=plot_color))
        fig.add_trace(go.Scatter(x=[x[0, 0]], y=[x[0, 1]], mode='markers',
             name='Start', marker=dict(color='blue', size=8)))
    return fig

def plot_env(x=None, u=None, goal=None, obs=None, boundary_wall=None):
    fig = go.Figure()
    obs1_x, obs1_y, obs1_w, obs1_h = obs[0]
    obs2_x, obs2_y, obs2_w, obs2_h = obs[1]
    fig.add_shape(type='rect', x0=obs1_x - obs1_w / 2,
                  y0=obs1_y - obs1_h / 2, x1=obs1_x + obs1_w / 2,
                  y1=obs1_y + obs1_h / 2, fillcolor='red', opacity=0.5)
    fig.add_shape(type='rect', x0=obs2_x - obs2_w / 2,
                  y0=obs2_y - obs2_h / 2, x1=obs2_x + obs2_w / 2,
                  y1=obs2_y + obs2_h / 2, fillcolor='red', opacity=0.5)
    fig.update_layout(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        width=800,
        height=800,
        title='Trajectory',
        showlegend=False,
        legend=dict(x=0, y=1),
        paper_bgcolor="LightSteelBlue",
        yaxis_range=[-2.5, 2.5],
        xaxis_range=[-3.5, 2.5]
    )
    return fig

def mppi_control(u, weights, device):
    # u: [N, num_steps, control_dim]
    # weights: [N]
    # u_new: [N, control_dim]
    w = torch.zeros(u.shape, device=device)
    w[:, :, :] = weights.unsqueeze(1).unsqueeze(2)
    u_new = torch.sum(w * u, dim=0) / torch.sum(w, dim=0)
    return u_new    

def run_batch_mppi(env, device, x0, dynamics, total_time_steps,
                   num_rollouts, planning_horizon, dt, goal, obs,
                   boundary_wall, softmax_lambda):
    START_TIME = time.time()
    goalX, goalY, goalR = goal
    running_costs = torch.zeros(total_time_steps, device=device)
    traj = [x0]
    control_traj = []
    mean_u = torch.zeros(num_rollouts, planning_horizon,
                         x0.shape[1], device=device)
    if env is None:
        env = plot_env(obs=obs)
    for step in range(0, total_time_steps):
        u = torch.normal(mean_u)
        def normalize_samples(samples):
            magnitude = torch.norm(samples, dim=-1)
            mask = magnitude > 1
            normalized_samples = torch.where(mask.unsqueeze(-1),
                         samples / magnitude.unsqueeze(-1), samples)
            return normalized_samples
        u = normalize_samples(u)
        current_x = traj[-1]
        running_cost = math.sqrt((current_x[0,0] - goalX)**2 + \
                                 (current_x[0, 1] - goalY)**2)
        running_costs[step] = running_cost
        next_x = batch_rollout_traj(current_x, u, dynamics, dt, device)
        cost = mppi_cost_func(next_x, None, goal, obs, boundary_wall,
                              device)
        best_u = mppi_control(u, torch.softmax(-softmax_lambda * cost,
                              dim=0), device)
        n_steps = min(planning_horizon, total_time_steps - step - 1)
        mean_u = best_u[0:n_steps, :]
        mean_u = mean_u.repeat(num_rollouts, 1, 1)
        best_x = dyn_propogate(current_x, best_u[0, :].reshape(1, 2),
                               dynamics, dt, device)
        traj.append(best_x)
        control_traj.append(best_u[0, :].reshape(1, 2))

    traj = torch.stack(traj, dim=1).reshape(total_time_steps + 1, 2)
    control_traj = torch.stack(control_traj, dim=1).\
                        reshape(total_time_steps, 2)
    env = plot_traj(env, traj)
    # print(f"Time taken by FUNC is {time.time() - START_TIME:.2f}")
    return torch.trapezoid(running_costs, dx=dt), env, traj \
                    , control_traj, time.time() - START_TIME 

def get_matlab_variables(mat_file_path):
    variables = loadmat(mat_file_path)
    v1_data = variables['data_v1']
    variables['deriv_v1'] = variables['deriv_v1'].squeeze()
    deriv_v1 = []
    deriv_v1.append(variables['deriv_v1'][0])
    deriv_v1.append(variables['deriv_v1'][1])
    deriv_v1.append(variables['deriv_v1'][2])
    opt_ctrl_v1 = []
    variables['opt_ctrl_v1'] = variables['opt_ctrl_v1'].squeeze()
    opt_ctrl_v1.append(variables['opt_ctrl_v1'][0])
    opt_ctrl_v1.append(variables['opt_ctrl_v1'][1])
    variables['grid'] = variables['grid'].squeeze()
    coordinates = variables['grid'].item()[6].squeeze()
    x_coord = coordinates[0].squeeze()
    y_coord = coordinates[1].squeeze()
    tau = variables['tau'].squeeze()
    matlab_var_dict = dict(
        v1_data=v1_data,
        deriv_v1=deriv_v1,
        opt_ctrl_v1=opt_ctrl_v1,
        x_coord=x_coord,
        y_coord=y_coord,
        tau=tau
    )
    return matlab_var_dict

def get_deriv_v1_eval_func(matlab_var_dict):
    eval_deriv_v1_x = RegularGridInterpolator(( matlab_var_dict['x_coord'], \
        matlab_var_dict['y_coord'], matlab_var_dict['tau']), \
        matlab_var_dict['deriv_v1'][0] ,  bounds_error=False, fill_value=None)
    eval_deriv_v1_y = RegularGridInterpolator(( matlab_var_dict['x_coord'], \
        matlab_var_dict['y_coord'], matlab_var_dict['tau']), \
        matlab_var_dict['deriv_v1'][1] ,  bounds_error=False, fill_value=None)
    eval_deriv_v1_t = RegularGridInterpolator(( matlab_var_dict['x_coord'], \
        matlab_var_dict['y_coord'], matlab_var_dict['tau']), \
        matlab_var_dict['deriv_v1'][2] ,  bounds_error=False, fill_value=None)
    return eval_deriv_v1_x, eval_deriv_v1_y, eval_deriv_v1_t

def get_opt_ctrl_v1_eval_func(matlab_var_dict):
    eval_opt_ctrl_v1_x = RegularGridInterpolator(( matlab_var_dict['x_coord'], \
        matlab_var_dict['y_coord'], matlab_var_dict['tau']), \
        matlab_var_dict['opt_ctrl_v1'][0] ,  bounds_error=False, fill_value=None)
    eval_opt_ctrl_v1_y = RegularGridInterpolator(( matlab_var_dict['x_coord'], \
        matlab_var_dict['y_coord'], matlab_var_dict['tau']), \
        matlab_var_dict['opt_ctrl_v1'][1] ,  bounds_error=False, fill_value=None)
    return eval_opt_ctrl_v1_x, eval_opt_ctrl_v1_y

def get_v1_data_eval_func(matlab_var_dict):
    eval_v1_data = RegularGridInterpolator(( matlab_var_dict['x_coord'], \
        matlab_var_dict['y_coord'], matlab_var_dict['tau']), \
        matlab_var_dict['v1_data'] ,  bounds_error=False, fill_value=None)
    return eval_v1_data

def get_LR_ctrl(x_with_t, eval_opt_ctrl_x, eval_opt_ctrl_y, device):
    u1 = eval_opt_ctrl_x(x_with_t)
    u2 = eval_opt_ctrl_y(x_with_t)
    u_opt = torch.tensor([u1, u2], device=device).reshape(1, 2)
    return u_opt

def get_QP_ctrl(x_with_t, u_nom_dev, eval_deriv_v1_x, eval_deriv_v1_y,
                eval_deriv_v1_t, dynamics, device):
    u_nom = u_nom_dev
    u_nom = u_nom.cpu()
    y = x_with_t[0,1].item()
    Vx = eval_deriv_v1_x(x_with_t)[0]
    Vy = eval_deriv_v1_y(x_with_t)[0]
    Vt = eval_deriv_v1_t(x_with_t)[0]
    u = cp.Variable(2)
    constraints = [cp.norm(u, 2) <= 1]
    Objective = cp.Maximize(Vt + Vx * u[0] + Vy * u[1] \
                            + Vx * (2 - 0.5*y*y ))
    prob = cp.Problem(Objective, constraints)
    best_constraint = prob.solve()
    best_constraint = min(best_constraint - 1e-5, 0)
    u = cp.Variable(2)
    constraints = [cp.norm(u, 2) <= 1, Vt + Vx * u[0] + Vy * u[1] \
                   + Vx * (2 - 0.5*y*y )>= best_constraint]
    Objective = cp.Minimize((u[0] - u_nom[0,1])**2 + (u[1] - u_nom[0,1])**2)
    prob = cp.Problem(Objective, constraints)
    result = prob.solve()
    u = u.value
    u_opt = torch.tensor([u[0], u[1]], device=device).reshape(1, 2)
    return u_opt    

def run_batch_mppi_qp(env, device, x0, dynamics, total_time_steps,
                      num_rollouts, planning_horizon, dt, goal, obs,
                      boundary_wall, softmax_lambda, eval_deriv_v1_x,
                      eval_deriv_v1_y, eval_deriv_v1_t, eval_v1_data,
                      eval_opt_ctrl_x, eval_opt_ctrl_y, v1_threshold):
    START_TIME = time.time()
    running_costs = torch.zeros(total_time_steps, device=device)
    traj = [x0]
    switching_traj = []
    ctrl_traj = []
    goalX, goalY, goalR = goal
    mean_u = torch.zeros(num_rollouts, planning_horizon, 2, device=device)
    if env is None:
        env = plot_env(obs=obs)
    for step in range(0, total_time_steps):
        n_steps = total_time_steps - step
        u = torch.normal(mean_u)
        def normalize_samples(samples):
            magnitude = torch.norm(samples, dim=-1)
            mask = magnitude > 1
            normalized_samples = torch.where(mask.unsqueeze(-1),
                                 samples / magnitude.unsqueeze(-1), samples)
            return normalized_samples
        u = normalize_samples(u)
        current_x = traj[-1]
        running_cost = math.sqrt((current_x[0,0] - goalX)**2 \
                                + (current_x[0, 1] - goalY)**2)
        running_costs[step] = running_cost
        next_x = batch_rollout_traj(current_x, u, dynamics, dt, device)
        cost = mppi_cost_func(next_x, None, goal, obs,
                               boundary_wall, device)
        best_u = mppi_control(u, torch.softmax(-softmax_lambda * cost,
                                     dim=0), device)
        mean_u = best_u[0:planning_horizon, :]
        mean_u = mean_u.repeat(num_rollouts, 1, 1)
        u_opt = best_u[0, :].reshape(1, 2)
        x_with_t = torch.tensor([[current_x[0, 0], current_x[0, 1],
                                  n_steps * dt]], device=device)
        x_with_t = x_with_t.cpu()
        v1 = eval_v1_data(x_with_t)
        if v1 < v1_threshold:
            u_opt = get_QP_ctrl(x_with_t, u_opt, eval_deriv_v1_x,
                                eval_deriv_v1_y, eval_deriv_v1_t, dynamics, device)
            # u_opt = get_LR_ctrl(x_with_t, eval_opt_ctrl_x,
                                # eval_opt_ctrl_y, device)
        best_x = dyn_propogate(current_x, u_opt, dynamics, dt, device)
        traj.append(best_x)
        if v1 < v1_threshold:
            switching_traj.append(best_x)
        ctrl_traj.append(u_opt)
    traj = torch.stack(traj, dim=1).reshape(total_time_steps + 1, 2)
    if len(switching_traj) > 0:
        switching_traj = torch.stack(switching_traj, dim=1).reshape(len(switching_traj), 2)
        env = plot_traj(env, switching_traj, 'red')
    ctrl_traj = torch.stack(ctrl_traj, dim=1).reshape(total_time_steps, 2)
    env = plot_traj(env, traj)
    return torch.trapezoid(running_costs, dx=dt), env, traj, ctrl_traj \
            , switching_traj, time.time() - START_TIME

