function uOpt = optCtrl(obj, t, x, deriv, schemeData)
    % u{1} = ux = velocity in x direction
    % u{2} = uy = velocity in y direction

    %% Input processing
    % if nargin < 5
    %   uMode = 'min';
    % end
    % ONLY DOING BRT AVOID 
    % This system is for the second stage only
    if ~iscell(deriv)
        deriv = num2cell(deriv);
    end
    if ~iscell(x)
        x = num2cell(x);
    end
    uOpt = cell(obj.nu, 1);
    f = obj.dyn_func_handle;
    control_bound_func_handle = obj.control_bound_func_handle;

    if size(x{1}, 1) == 1 && size(x{1}, 2) == 1
        if iscell(x)
            x = cell2mat(x);
        end
        % When called over a single point
        current_data_v1 = eval_u(schemeData.grid_with_time, schemeData.v1_data, [x',t]);
        current_deriv_v2 = cell2mat(deriv); 
        if current_data_v1 > schemeData.v1_threshold
            model.A =sparse(1, 2);
            model.rhs = 0;
            model.quadcon(1).Qc = sparse([1 0; 0 1]);
            model.quadcon(1).q = [0 0];
            model.quadcon(1).rhs = 1;
            model.obj = [current_deriv_v2(1); current_deriv_v2(2)];
            model.lb = [-1; -1];
            model.modelsense = 'min';
            params.OutputFlag = 0;
            result = gurobi(model, params);
            if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
                u_opt = result.x;
                uOpt{1} = u_opt(1);
                uOpt{2} = u_opt(2);
            else
                error('The following error occurred outside brt: %s', result.status);
            end
        else
            current_deriv_v1 = eval_u(schemeData.grid_with_time, schemeData.deriv_v1, [x', t]);
            model.A = sparse(1,2);
            model.rhs = [0];
            model.quadcon(1).Qc = sparse([1 0; 0 1]);
            model.quadcon(1).q = [0; 0];
            model.quadcon(1).rhs = 1;
            model.obj = [current_deriv_v1(1); current_deriv_v1(2)];
            model.modelsense = 'max';
            model.lb = [-1; -1];
            model.sense = '=';
            params.OutputFlag = 0;
            result = gurobi(model, params);
            if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
                best_constraint = -current_deriv_v1(3) + current_deriv_v1(1:2)' * result.x ...
                                    + current_deriv_v1(1) * (2 - 0.5 * x(2) * x(2));
                best_constraint = min(best_constraint, 0);
            else
                error('The following error occurred in best_cons: %s', result.status);
            end

            model.A = sparse([current_deriv_v1(1)  current_deriv_v1(2)]);
            model.rhs = best_constraint + current_deriv_v1(3) - current_deriv_v1(1) * (2 - 0.5 * x(2) * x(2));
            model.quadcon(1).Qc = sparse([1 0; 0 1]);
            model.quadcon(1).q = [0 0];
            model.quadcon(1).rhs = 1;
            model.obj = [current_deriv_v2(1); current_deriv_v2(2)];
            model.modelsense = 'min';
            model.lb = [-1; -1];
            model.sense = '>';
            params.OutputFlag = 0;
            result = gurobi(model, params);
            if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
                u_opt = result.x;
                uOpt{1} = u_opt(1);
                uOpt{2} = u_opt(2);
            else
                error('The following error occurred in cooptimization: %s', result.status);
            end
        end
    else % When x is a grid 
        if size(deriv{1}, 1) == 1 && size(deriv{1}, 2) == 1 % account for calls from genericPartial
            deriv{1} = ones(size(x{1})) .* deriv{1};
            deriv{2} = ones(size(x{2})) .* deriv{2};
        end
        
        [~, v1_idx] = min(abs(t - schemeData.tau));
    
        current_deriv_x = schemeData.deriv_v1{1}(:,:,v1_idx);
        current_deriv_y = schemeData.deriv_v1{2}(:,:,v1_idx);
        current_deriv_t = schemeData.deriv_v1{3}(:,:,v1_idx);
        state_y_flat = schemeData.grid.xs{2}(:);
        num_grid_points = schemeData.grid.N(1) * schemeData.grid.N(2);
        linear_idx_v1_leq0 = schemeData.linear_idx_v1_leq0{v1_idx};
        v1_x_flat = current_deriv_x(:);
        v1_y_flat = current_deriv_y(:);
        v1_t_flat = current_deriv_t(:);
        Arow = [linear_idx_v1_leq0 linear_idx_v1_leq0];
        Acol = [linear_idx_v1_leq0 linear_idx_v1_leq0 + num_grid_points];
        Aval = [v1_x_flat(linear_idx_v1_leq0) v1_y_flat(linear_idx_v1_leq0)];
        A = sparse(Arow, Acol, Aval, num_grid_points, 2*num_grid_points);
        b = zeros(num_grid_points, 1);
        b(linear_idx_v1_leq0) = schemeData.best_constraints{v1_idx}(linear_idx_v1_leq0) ...
                                    + v1_t_flat(linear_idx_v1_leq0) ...
                                - v1_x_flat(linear_idx_v1_leq0) .* ... 
                                (2 - 0.5 * state_y_flat(linear_idx_v1_leq0) .* state_y_flat(linear_idx_v1_leq0));
        c = [deriv{1}(:); deriv{2}(:)];
        
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
        model.lb = -1 * ones(2*num_grid_points, 1);
        model.obj = c;
        model.rhs = b;
        model.sense = '>';
        model.modelsense = 'min';
        params.OutputFlag = 0;
        result = gurobi(model, params);
        if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
            u_opt_stacked = result.x;
        else
            error('The following error occurred: %s', result.status);
        end
        
        uOpt{1} = reshape(u_opt_stacked(1:num_grid_points), [schemeData.grid.N(1), schemeData.grid.N(2)]);
        uOpt{2} = reshape(u_opt_stacked(num_grid_points + 1:2*num_grid_points), [schemeData.grid.N(1), schemeData.grid.N(2)]);
    end
    % check if opt ctrl is unbounded
    if any(isnan(uOpt{1}))
        error('opt ctrl is unbounded')
    end
    if any(isnan(uOpt{2}))
        error('opt ctrl is unbounded')
    end
end
