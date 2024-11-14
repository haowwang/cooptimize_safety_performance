function [state_traj, ctrl_traj, value_traj, value_safety_traj] = simulateTrajectoryCtrlConstrained(grid, ...
    dyn_sys, data_v1, data_v2, deriv_v1, deriv_v2, opt_ctrl_v1, initial_state, ...
    tau, dt, v1_threshold, traj_idx, goal_condition, gamma, use_cbf_constraint)
% note: deriv_v1 is taken with respect to time in addition to state
% Update Notes
% 8/19/24 - cleaned up to work with the boat2d brt oc rollout test script. 

    tInit = tau(1);
    tMax = tau(end);
    t = tMax; 
    i = 1;
    state_traj(:,1) = initial_state; 
    grid_with_time = createGrid([grid.min; tau(1)], [grid.max; tau(end)], [grid.shape, length(tau)]);
    while t > tInit + 1e-6
        fprintf('Traj %d Ctrl Cons Formulation Rollout t = %0.3f \n',traj_idx, t);

        current_state = state_traj(:, i); 
        current_deriv_v2 = eval_u(grid_with_time, deriv_v2, [current_state', t]); % array, not a cell
        
        value_traj(i) = eval_u(grid_with_time, data_v2, [current_state', t]);
        value_safety_traj(i) = eval_u(grid_with_time, data_v1, [current_state', t]);

        extra_args.dynSys = dyn_sys;
        extra_args.opt_v1_control = opt_ctrl_v1;
        extra_args.tau = tau; 
        extra_args.grid = grid;
        extra_args.v1_data = data_v1; 
        extra_args.v1_threshold = v1_threshold; 
        extra_args.deriv_v1 = deriv_v1; 
        extra_args.uMode = 'min'; 
        extra_args.grid_with_time = grid_with_time; 
        extra_args.gamma = gamma;
        extra_args.use_cbf_constraint = use_cbf_constraint;

        current_opt_ctrl = dyn_sys.optCtrlRollOut(t, current_state, current_deriv_v2, extra_args); % cell array
        current_state_dot = dyn_sys.dynamics(0, num2cell(current_state), current_opt_ctrl, ...
              [], extra_args); % cell array
            
        next_state = current_state + dt * cell2mat(current_state_dot)'; 
        state_traj(:, i+1) = next_state; 
        ctrl_traj(:, i) = cell2mat(current_opt_ctrl); 

        if ~isempty(goal_condition)
            if goal_condition(next_state(1:2,:)) 
              return
            end
        end
            
        t = t - dt; 
        i = i + 1; 

    end

end

%% 


