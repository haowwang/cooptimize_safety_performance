function [data] = computeCtrlConsValueFunc(grid, dyn_sys, tau, v1_mode, ...
                                v1_data, deriv_v1, opt_v1_control, ...
                                running_cost, terminal_cost, compute_best_constraints, ...
                                use_analytical_optimal_control, accuracy, v1_threshold)
    % running_cost: function handle which takes in the current state and control and returns the running cost
    % terminal_cost: function handle which takes in the current state and returns the terminal cost 
    % A non-zero terminal cost is needed for the value function derivatives to be non-zero at the terminal time
    % dyn_sys should have the following function handles as properties: 
    %   - dyn_func_handle: function handle which takes in the state and control and returns the dynamics
    %   - control_bound_func_handle: function handle which takes in the control and returns the constraint

    constraint_epsilon = 1e-4;
    num_grid_points = grid.N(1) * grid.N(2);

    if strcmp(v1_mode, 'brt_reach')
        % Use opt safe control for 0 < v1 (v1_g0)
        % Optimize over live controls for -v1_threshold <= v1 <= 0 (v1_eq0)
        % Optimize over all controls for v1 < -v1_threshold (v1_l0)
        uMode = 'min';
        dMode = 'max'; 
    elseif strcmp(v1_mode, 'brt_avoid')
        % Optimize over all controls for v1_threshold < v1 (v1_g0)
        % Optimize over safe controls for 0 <= v1 <= v1_threshold (v1_eq0)
        % Use opt safe control for v1 < 0 (v1_g0)
        uMode = 'max';
        dMode = 'min';
        v1_l0_mask = (v1_data < 0);
        v1_eq0_mask = (v1_data <= v1_threshold & v1_data >= 0);
        v1_g0_mask = (v1_data > v1_threshold);

    elseif strcmp(v1_mode, 'brat')
        % Use opt safe control for 0 < v1 (v1_g0)
        % Optimize over live controls for -v1_threshold <= v1 <= 0 (v1_eq0)
        % Optimize over all controls for v1 < -v1_threshold (v1_l0)
        uMode = 'min';
        dMode = 'max';        
        v1_g0_mask = (v1_data > 0);
        v1_eq0_mask = (v1_data >= -v1_threshold & v1_data <= 0);
        v1_l0_mask = (v1_data < -v1_threshold);

    elseif strcmp(v1_mode, 'bras')
        % Use opt safe control for 0 < v1 (v1_g0)
        % Optimize over live controls for -v1_threshold <= v1 <= 0 (v1_eq0)
        % Optimize over all controls for v1 < -v1_threshold (v1_l0)
        uMode = 'min';
        dMode = 'max'; 
    else 
        error('Unknown v1_mode');
    end

    for i = 1:length(tau)
        [v1_l0_subscript_x{i}, v1_l0_subscript_y{i}] = ...
            ind2sub([grid.N(1), grid.N(2)], find(v1_l0_mask(:,:,i)));
        [v1_eq0_subscript_x{i}, v1_eq0_subscript_y{i}] = ...
            ind2sub([grid.N(1), grid.N(2)], find(v1_eq0_mask(:,:,i)));
        [v1_g0_subscript_x{i}, v1_g0_subscript_y{i}] = ...
            ind2sub([grid.N(1), grid.N(2)], find(v1_g0_mask(:,:,i)));
        current_v1_data = v1_data(:,:,i);
        linear_idx_v1_leq0{i} = find(current_v1_data(:) <= 0);
        linear_idx_v1_g0{i} = find(current_v1_data(:) > 0);
    end 

    if compute_best_constraints
        for i = 1:length(tau)
            current_deriv_x = deriv_v1{1}(:,:,i);
            current_deriv_y = deriv_v1{2}(:,:,i);
            current_deriv_t = deriv_v1{3}(:,:,i);

            A = sparse(num_grid_points, 2 * num_grid_points);
            b = zeros(num_grid_points, 1);
            v1_x_flat = current_deriv_x(:);
            v1_y_flat = current_deriv_y(:);
            v1_t_flat = current_deriv_t(:);
            c = [v1_x_flat; v1_y_flat];
            Qrow = [(1:num_grid_points).', (1:num_grid_points).' + num_grid_points];
            Qcol = [(1:num_grid_points).', (1:num_grid_points).' + num_grid_points];
            Qval = ones(num_grid_points, 2);
            q = sparse(2*num_grid_points, 1);
            rhs = 1;
            model.quadcon = struct('Qrow', num2cell(Qrow, 2), ...
                                  'Qcol', num2cell(Qcol, 2), ...
                                  'Qval', num2cell(Qval, 2), ...
                                  'q', repmat({q}, num_grid_points, 1), ...
                                  'rhs', repmat(rhs, 1));

            model.A = A;
            model.rhs = b;
            model.obj = c;
            model.modelsense = 'max';
            model.lb = - ones(2 * num_grid_points, 1);
            model.sense = '=';
            params.OutputFlag = 0;
            result = gurobi(model, params);
            if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
                u_best_stacked = result.x;
            else
                error('The following error occurred: %s', result.status);
            end
            best_constraints{i} = -v1_t_flat + v1_x_flat .* u_best_stacked(1:num_grid_points) ...
                                            + v1_y_flat .* u_best_stacked(num_grid_points + 1:2*num_grid_points) ...
                                            + v1_x_flat .* (2 - 0.5 * grid.xs{2}(:) .* grid.xs{2}(:));

            best_constraints{i} = min(best_constraints{i}, 0);
            fprintf(' %d out of %d \n', i,length(tau));
        end
    else 
        for i = 1:length(tau)
            best_constraints_eq0{i} = zeros(length(v1_eq0_subscript_x{i}));
            best_constraints{i} = zeros(num_grid_points, 1);
        end
    end

    schemeData.best_constraints = best_constraints;
    schemeData.linear_idx_v1_leq0 = linear_idx_v1_leq0;
    schemeData.linear_idx_v1_g0 = linear_idx_v1_g0;
    schemeData.use_analytical_optimal_control = use_analytical_optimal_control;

    schemeData.dynSys = dyn_sys;
    schemeData.grid = grid; 
    schemeData.tau = tau;
    schemeData.hamFunc = @genericCoopHam;
    schemeData.uMode = uMode; 
    schemeData.dMode = dMode; 
    schemeData.tMode = 'backward';
    schemeData.accuracy = accuracy;
    
    schemeData.deriv_v1 = deriv_v1;
    schemeData.opt_v1_control = opt_v1_control;
    schemeData.v1_data = v1_data;
    schemeData.v1_threshold = v1_threshold;
    schemeData.constraint_epsilon = constraint_epsilon;
    schemeData.v1_mode = v1_mode;
    
    %% Start dynamic programming from the terminal value function
    HJIextraArgs.visualize.initialValueFunction = 1;
    HJIextraArgs.visualize.valueFunction = 1;
    HJIextraArgs.visualize.figNum = 1; %set figure number
    HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
    HJIextraArgs.visualize.xTitle = 'x position';
    HJIextraArgs.visualize.yTitle = 'y position';
    HJIextraArgs.visualize.zTitle = 'Value Function';
    HJIextraArgs.visualize.viewGrid = true;

    data0 = terminal_cost(grid.xs);
    tic
    [data, ~, ~] = HJIPDE_solve(data0, tau, schemeData, 'none', HJIextraArgs);
    disp('Second stage computation time: ')
    toc

end

