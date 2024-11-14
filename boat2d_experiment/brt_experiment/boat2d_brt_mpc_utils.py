import casadi as ca
import plotly.graph_objects as go
import numpy as np 
import math
import time

def signed_distance_to_rectangle_casadi(point, rectangle):
    cx, cy = rectangle[0], rectangle[1]
    rect_width = rectangle[2]
    rect_height = rectangle[3]
    half_width = rect_width / 2.0
    half_height = rect_height / 2.0
    
    left = cx - half_width
    right = cx + half_width
    bottom = cy - half_height
    top = cy + half_height

    dx = ca.fmax(ca.fmax(left - point[0], point[0] - right), 0)
    dy = ca.fmax(ca.fmax(bottom - point[1], point[1] - top), 0)
    
    outside_distance = ca.sqrt(dx**2 + dy**2)
    inside_distance_x = ca.fmin(point[0] - left, right - point[0])
    inside_distance_y = ca.fmin(point[1] - bottom, top - point[1])
    inside_distance = ca.fmin(inside_distance_x, inside_distance_y)
    inside = (left <= point[0]) * (point[0] <= right) * (bottom <= point[1]) * (point[1] <= top)
    signed_distance = ca.if_else(inside, -inside_distance, outside_distance)
    return signed_distance

def mpc_cost_func(x, u, goal, obs, boundary_wall):
    goalX, goalY, goalR = goal
    goal_dist = ca.sqrt((x[0] - goalX)**2 + (x[1] - goalY)**2)
    cost = goal_dist 
    return cost

def dyn_propogate(x, u, dt, dynamics):
    dx = dynamics(x, u)
    return [x[0] + dt* dx[0], x[1] + dt * dx[1]] 

def get_mpc_ctrl(x0, horizon, goal, obs, boundary_wall, init_guess, dt, dynamics):
    states = ca.MX.sym('state', horizon, 2)
    ctrls = ca.MX.sym('ctrl', horizon - 1, 2)
    g = [states[0, 0] - x0[0], states[0, 1] - x0[1]]
    lbg = [0, 0]
    ubg = [0, 0]
    cost = 0
    for i in range(horizon - 1):
        next_state = dyn_propogate(states[i, :], ctrls[i, :], dt, dynamics)
        g.append(states[i + 1, 0] - next_state[0])
        lbg.append(0)
        ubg.append(0)
        g.append(states[i + 1, 1] - next_state[1])
        lbg.append(0)
        ubg.append(0)
        g.append(ctrls[i, 0]**2 + ctrls[i, 1]**2)
        lbg.append(0)
        ubg.append(1)
        g.append(signed_distance_to_rectangle_casadi(next_state, obs[0]))
        lbg.append(0)
        ubg.append(ca.inf)
        g.append(signed_distance_to_rectangle_casadi(next_state, obs[1]))
        lbg.append(0)
        ubg.append(ca.inf)
        g.append(next_state[0] - boundary_wall[0])
        lbg.append(0)
        ubg.append(ca.inf)
        g.append(next_state[0] - boundary_wall[1])
        lbg.append(-ca.inf)
        ubg.append(0)
        g.append(next_state[1] - boundary_wall[2])
        lbg.append(0)
        ubg.append(ca.inf)
        g.append(next_state[1] - boundary_wall[3])
        lbg.append(-ca.inf)
        ubg.append(0)
        cost += mpc_cost_func(states[i, :], ctrls[i, :], goal, obs, boundary_wall)
    g = ca.vertcat(*g)
    opts = {
    "print_time": False,     
    "ipopt": {
        "print_level": 0,
        "max_cpu_time": 4,
        },
    }
    states = ca.reshape(states, -1, 1)
    ctrls = ca.reshape(ctrls, -1, 1)
    solver = ca.nlpsol('solver', 'ipopt', {'x': ca.vertcat(states, ctrls), 'f': cost, 'g': g}, opts)
    res = solver(x0=init_guess, lbg=lbg, ubg=ubg)
    u_best0 = res['x'][2*horizon].full().flatten()[0]
    u_best1 = res['x'][3*horizon - 1].full().flatten()[0]
    return [u_best0, u_best1], res['x'].full().flatten().tolist()

def plot_traj(fig, x, plot_color='blue'):
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

def run_mpc_rollout(x0, goal, obs, boundary_wall, horizon, dt, total_time_steps, dynamics, env = None):
    state_traj = [x0]
    ctrl_traj = []
    solver_init = [0] * (2 * (2 * horizon - 1))
    if env is None:
        env = plot_env(obs=obs)
    running_costs = np.zeros(total_time_steps)
    start_time = time.time()
    for i in range(total_time_steps):
        iter_start = time.time()
        current_horizon = min(horizon, total_time_steps - i + 1)
        current_init_solver = solver_init[1:current_horizon] + [solver_init[0]]
        current_init_solver += solver_init[current_horizon+1:current_horizon + current_horizon] + [solver_init[current_horizon]]
        current_init_solver += solver_init[2 * current_horizon + 1:2 * current_horizon + current_horizon - 1] + [solver_init[2 * current_horizon]]
        current_init_solver += solver_init[3 * current_horizon :3 * current_horizon - 1 + current_horizon - 1] + [solver_init[3 * current_horizon - 1]]
        current_state = state_traj[-1]
        running_cost = math.sqrt((current_state[0] - goal[0])**2 + \
                                 (current_state[1] - goal[1])**2)
        running_costs[i] = running_cost
        u_mpc, solver_init = get_mpc_ctrl(current_state, current_horizon, goal, obs,
                          boundary_wall, current_init_solver, dt, dynamics)
        next_state = dyn_propogate(current_state, u_mpc, dt, dynamics)
        state_traj.append(next_state)
        ctrl_traj.append(u_mpc)
        print(f"Time step {i+1} of {total_time_steps} took {time.time() - iter_start} seconds")
    time_taken = time.time() - start_time
    state_traj = np.array(state_traj)
    ctrl_traj = np.array(ctrl_traj)
    env = plot_traj(env, state_traj)
    return np.trapz(running_costs), state_traj, ctrl_traj, time_taken, env