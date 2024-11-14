function u_opt = computeCtrlConsOptCtrl(grid, dyn_sys, tau, v1_mode, v1_data, deriv_v1, running_cost, v2_data)
    u_opt{1} = zeros(grid.N(1), grid.N(2), length(tau));
    u_opt{2} = zeros(grid.N(1), grid.N(2), length(tau));
    f = dyn_sys.dyn_func_handle;
    L = running_cost;
    control_bound_func_handle = dyn_sys.control_bound_func_handle;
    v1_threshold = 0.05;
    grid_with_time = createGrid([grid.min; tau(1)], [grid.max; tau(end)], [grid.shape, length(tau)]);
    deriv = computeGradients(grid_with_time, v2_data);
    if strcmp(v1_mode, 'brt_reach')
        error('Not implemented yet');
    elseif strcmp(v1_mode, 'brt_avoid')
        error('Not implemented yet');
    elseif strcmp(v1_mode, 'brat')
        v1_l0_mask = (v1_data < -v1_threshold);
        v1_eq0_mask = (v1_data >= -v1_threshold & v1_data <= 0);
        v1_g0_mask = (v1_data > 0);
        
        for i = 1:length(tau)
            [v1_l0_subscript_x{i}, v1_l0_subscript_y{i}] = ...
                ind2sub([grid.N(1), grid.N(2)], find(v1_l0_mask(:,:,i)));
            [v1_eq0_subscript_x{i}, v1_eq0_subscript_y{i}] = ...
                ind2sub([grid.N(1), grid.N(2)], find(v1_eq0_mask(:,:,i)));
            [v1_g0_subscript_x{i}, v1_g0_subscript_y{i}] = ...
                ind2sub([grid.N(1), grid.N(2)], find(v1_g0_mask(:,:,i)));
        end        

        for i = 1:length(tau)
            best_constraints{i} = zeros(length(v1_g0_subscript_x{i})); 
            yalmip('clear');
            u_best = sdpvar(length(v1_g0_subscript_x{i}), 2);
            Constraints = [];
            Objective = 0;
            for j = 1:length(v1_g0_subscript_x{i})
                xIdx = v1_g0_subscript_x{i}(j);
                yIdx = v1_g0_subscript_y{i}(j);
                state = [grid.xs{1}(xIdx, yIdx); grid.xs{2}(xIdx, yIdx)];
                deriv_v1_at_point = [deriv_v1{1}(xIdx, yIdx, i); ...
                    deriv_v1{2}(xIdx, yIdx, i); ...
                    deriv_v1{3}(xIdx, yIdx, i)];
                Constraints = [Constraints, dyn_sys.control_bound_func_handle(u_best(j,:))];
                Objective = Objective + deriv_v1_at_point(end) + ...
                    deriv_v1_at_point(1:2)' * dyn_sys.dyn_func_handle(state, u_best(j,:));
            end
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            optimize(Constraints, Objective, Options);
            for j = 1:length(v1_g0_subscript_x{i})
                xIdx = v1_g0_subscript_x{i}(j);
                yIdx = v1_g0_subscript_y{i}(j);
                state = [grid.xs{1}(xIdx, yIdx); grid.xs{2}(xIdx, yIdx)];
                deriv_v1_at_point = [deriv_v1{1}(xIdx, yIdx, i); ...
                    deriv_v1{2}(xIdx, yIdx, i); ...
                    deriv_v1{3}(xIdx, yIdx, i)];
                objective = deriv_v1_at_point(end) + ...
                    deriv_v1_at_point(1:2)' * dyn_sys.dyn_func_handle(state, u_best(j,:));
                    objective = objective + 0.01;
                    best_constraints{i}(j) = max(objective, 0);
            end
            fprintf(' %d out of %d  %f\n', i,length(tau), max(best_constraints{i}, [], 'all'));
        end
        for time_idx = 1:length(tau)
            disp(time_idx);
            v1_idx = time_idx;
            %%%%%%%%%%%%%%%% For v1 > 0 %%%%%%%%%%%%%%%%
            yalmip('clear');
            num_states_v1_g0 = length(v1_g0_subscript_x{v1_idx});
            u_v1_g0 = sdpvar(num_states_v1_g0, 2);
            Constraints = [];
            Objective = 0;
            for i = 1:num_states_v1_g0
                xIdx = v1_g0_subscript_x{v1_idx}(i);
                yIdx = v1_g0_subscript_y{v1_idx}(i);
                state = [grid.xs{1}(xIdx, yIdx); grid.xs{2}(xIdx, yIdx)];
                deriv_v1_at_point = [deriv_v1{1}(xIdx, yIdx, v1_idx); ...
                    deriv_v1{2}(xIdx, yIdx, v1_idx); ...
                    deriv_v1{3}(xIdx, yIdx, v1_idx)];
                deriv_at_point = [deriv{1}(xIdx, yIdx, v1_idx); deriv{2}(xIdx, yIdx, v1_idx)];
                Constraints = [Constraints, ...
                    control_bound_func_handle(u_v1_g0(i,:))];
                Constraints = [Constraints, ...
                    deriv_v1_at_point(end) + deriv_v1_at_point(1:2)' * f(state, u_v1_g0(i,:)) ...
                    <= best_constraints{v1_idx}(i)];
                Objective = Objective + (L(state, u_v1_g0(i,:)) + deriv_at_point' * f(state, u_v1_g0(i,:)));
            end
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            sol = optimize(Constraints, Objective, Options);
            if sol.problem == 0
                u_v1_g0 = value(u_v1_g0);
            else
                error('The following error occurred: %s', sol.info);
            end
            for i = 1:num_states_v1_g0
                xIdx = v1_g0_subscript_x{v1_idx}(i);
                yIdx = v1_g0_subscript_y{v1_idx}(i);
                u_opt{1}(xIdx, yIdx, v1_idx) = u_v1_g0(i, 1);
                u_opt{2}(xIdx, yIdx, v1_idx) = u_v1_g0(i, 2);
            end
            %%%%%%%%%%%%%%%% For v1 = 0 %%%%%%%%%%%%%%%%
            yalmip('clear');
            num_states_v1_eq0 = length(v1_eq0_subscript_x{v1_idx});
            u_v1_eq0 = sdpvar(num_states_v1_eq0, 2);
            Constraints = [];
            Objective = 0;
            for i = 1:num_states_v1_eq0
                xIdx = v1_eq0_subscript_x{v1_idx}(i);
                yIdx = v1_eq0_subscript_y{v1_idx}(i);
                state = [grid.xs{1}(xIdx, yIdx); grid.xs{2}(xIdx, yIdx)];
                deriv_v1_at_point = [deriv_v1{1}(xIdx, yIdx, v1_idx); ...
                    deriv_v1{2}(xIdx, yIdx, v1_idx); ...
                    deriv_v1{3}(xIdx, yIdx, v1_idx)];
                deriv_at_point = [deriv{1}(xIdx, yIdx, v1_idx); deriv{2}(xIdx, yIdx, v1_idx)];
                Constraints = [Constraints, ...
                    control_bound_func_handle(u_v1_eq0(i,:))];
                Constraints = [Constraints, ...
                    deriv_v1_at_point(end) + deriv_v1_at_point(1:2)' * f(state, u_v1_eq0(i,:)) <= 0];
                Objective = Objective + (L(state, u_v1_eq0(i,:)) + deriv_at_point' * f(state, u_v1_eq0(i,:)));
            end
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            sol = optimize(Constraints, Objective, Options);
            if sol.problem == 0
                u_v1_eq0 = value(u_v1_eq0);
            else
                error('The following error occurred: %s', sol.info);
            end
            for i = 1:num_states_v1_eq0
                xIdx = v1_eq0_subscript_x{v1_idx}(i);
                yIdx = v1_eq0_subscript_y{v1_idx}(i);
                u_opt{1}(xIdx, yIdx, v1_idx) = u_v1_eq0(i, 1);
                u_opt{2}(xIdx, yIdx, v1_idx) = u_v1_eq0(i, 2);
            end
            %%%%%%%%%%%%%%%% For v1 < 0 %%%%%%%%%%%%%%%%
            yalmip('clear');
            num_states_v1_l0 = length(v1_l0_subscript_x{v1_idx});
            u_v1_l0 = sdpvar(num_states_v1_l0, 2);
            Constraints = [];
            Objective = 0;
            for i = 1:num_states_v1_l0
                xIdx = v1_l0_subscript_x{v1_idx}(i);
                yIdx = v1_l0_subscript_y{v1_idx}(i);
                state = [grid.xs{1}(xIdx, yIdx); grid.xs{2}(xIdx, yIdx)];

                deriv_at_point = [deriv{1}(xIdx, yIdx, v1_idx); deriv{2}(xIdx, yIdx, v1_idx)];
                Constraints = [Constraints, ...
                    control_bound_func_handle(u_v1_l0(i,:))];
                Objective = Objective + (L(state, u_v1_l0(i,:)) + deriv_at_point' * f(state, u_v1_l0(i,:)));
            end
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            sol = optimize(Constraints, Objective, Options);
            if sol.problem == 0
                u_v1_l0 = value(u_v1_l0);
            else
                error('The following error occurred: %s', sol.info);
            end
            for i = 1:num_states_v1_l0
                xIdx = v1_l0_subscript_x{v1_idx}(i);
                yIdx = v1_l0_subscript_y{v1_idx}(i);
                u_opt{1}(xIdx, yIdx, v1_idx) = u_v1_l0(i, 1);
                u_opt{2}(xIdx, yIdx, v1_idx) = u_v1_l0(i, 2);
            end
        end
    elseif strcmp(v1_mode, 'bras')
        error('Not implemented yet');
    else
        error('Unknown v1_mode');
    end
end