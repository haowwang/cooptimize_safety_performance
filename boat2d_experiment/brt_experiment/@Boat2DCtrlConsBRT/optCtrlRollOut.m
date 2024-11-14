function uOpt = optCtrl(obj, t, x, deriv, schemeData)
    % u{1} = ux = velocity in x direction
    % u{2} = uy = velocity in y direction

    %% Input processing
    % if nargin < 5
    %   uMode = 'min';
    % end
    % ONLY DOING BRT AVOID 
    % This system is for the second stage only
    

    uOpt = cell(obj.nu, 1);

    current_deriv_v2 = deriv;

    current_data_v1 = eval_u(schemeData.grid_with_time, schemeData.v1_data, [x',t]);

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
        % compute max margin
        % yalmip('clear');
        % u_v1_eq0 = sdpvar(2, 1, 'full');
        % current_deriv_v1 = eval_u(schemeData.grid_with_time, schemeData.deriv_v1, [x', t]);
        % Constraints = [control_bound_func_handle(u_v1_eq0)];
        % Objective = -(-current_deriv_v1(3) + current_deriv_v1(1:2)' * f(x, u_v1_eq0)); % TODO: double check sign of time derivative
        % Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
        % sol = optimize(Constraints, Objective, Options);
        % if sol.problem == 0
        %     best_constraint = - value(Objective);
        %     best_constraint = min(best_constraint, 0);
        % else
        %     error('In max margin computation, the following error occurred: %s', sol.info);
        % end
        current_deriv_v1 = eval_u(schemeData.grid_with_time, schemeData.deriv_v1, [x', t]);
        model.A = sparse(1,2);
        model.rhs = 0;
        model.quadcon(1).Qc = sparse([1 0; 0 1]);
        model.quadcon(1).q = [0 0];
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
        
        % compute opt ctrl on brt boundary
        % yalmip('clear');
        % current_deriv_v2 = cell2mat(deriv); 
        % u_v1_eq0 = sdpvar(2, 1);
        % Constraints = [control_bound_func_handle(u_v1_eq0), ...
        %     -current_deriv_v1(3) + current_deriv_v1(1:2)' * f(x, u_v1_eq0) >=  best_constraint];
        % Objective = (L(x, u_v1_eq0) + current_deriv_v2' * f(x, u_v1_eq0));
        % Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
        % sol = optimize(Constraints, Objective, Options);
        % if sol.problem == 0
        %     uOpt = num2cell(value(u_v1_eq0));
        % else
        %     error('In opt ctrl computation, the following error occurred: %s', sol.info);
        % end
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
            [~, v1_idx] = min(abs(t - schemeData.grid_with_time.vs{3}));
            figure;
            hold on; 
            contour(schemeData.grid.xs{1}, schemeData.grid.xs{2}, schemeData.v1_data(:,:,v1_idx), [0,0]);
            scatter(x(1), x(2), 50); 
            keyboard;
      
            error('The following error occurred in cooptimization: %s', result.status);
        end
    end
end