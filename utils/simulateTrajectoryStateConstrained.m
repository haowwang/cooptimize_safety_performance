function [state_traj,ctrl_traj,aux_value_traj, value_traj] ...
    = simulateTrajectoryStateConstrained(grid_vfunc_aux, data_vfunc_aux, data_vfunc, ...
    deriv_vfunc_aux, dyn_sys, state_init, tau, dt, goal_condition, traj_idx)

    state_dim = grid_vfunc_aux.dim - 1; 
    grid_vfunc_with_time = createGrid([grid_vfunc_aux.min(1:state_dim); tau(1)], ...
        [grid_vfunc_aux.max(1:state_dim); tau(end)], [grid_vfunc_aux.shape(1:state_dim), length(tau)]);
    grid_vfunc_aux_with_time = createGrid([grid_vfunc_aux.min; tau(1)], ...
        [grid_vfunc_aux.max; tau(end)], [grid_vfunc_aux.shape, length(tau)]);
    
    tInit = tau(1);
    tMax = tau(end);
    t = tMax;
    
    % compute initial z
    z_init = eval_u(grid_vfunc_with_time, data_vfunc, [state_init',t]);
    state_traj(1:state_dim, 1) = state_init;
    state_traj(state_dim + 1, 1) = z_init;
    value_traj(1) = state_traj(grid_vfunc_aux.dim,1);
    aux_value_traj(1) = eval_u(grid_vfunc_aux_with_time, data_vfunc_aux, [state_traj(:,1)',t]);

    % rollout
    i = 1;
    while t > tInit + 1e-6
      fprintf('Traj %d State Cons Formulation Rollout t = %0.3f \n',traj_idx, t);

      current_state = state_traj(:, i); 

      % query value function gradient
      current_deriv_vfunc_aux = eval_u(grid_vfunc_aux_with_time, deriv_vfunc_aux, [current_state',t]);
        
      % Compute the optimal control
      args.uMode = 'min'; 
      args.use_analytical_opt_ctrl = false; 
      current_opt_ctrl = dyn_sys.optCtrl(0, current_state, current_deriv_vfunc_aux, args); 

      % compute current state dot 
      current_state_dot = dyn_sys.dynamics(0, num2cell(current_state), current_opt_ctrl, ...
          [], args);
      
      % Compute the next state. euler integration
      next_state = current_state + dt * cell2mat(current_state_dot); 
      
      if ~isempty(goal_condition)
        if goal_condition(next_state(1:2,:)) 
          return
        end
      end
      
      % record traj info
      state_traj(:, i+1) = next_state; 
      value_traj(i + 1) = eval_u(grid_vfunc_with_time, data_vfunc, [state_traj(1:2,i+1)',t-dt]); 
      aux_value_traj(i+1) = eval_u(grid_vfunc_aux_with_time, data_vfunc_aux, [state_traj(:,i+1)',t-dt]);
      ctrl_traj(:, i) = cell2mat(current_opt_ctrl);
     
      i = i+1;

      % Propagate time
      t = t - dt;
    end
end