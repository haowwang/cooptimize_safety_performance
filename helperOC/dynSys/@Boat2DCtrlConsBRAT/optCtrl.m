function uOpt = optCtrl(obj, t, x, deriv, schemeData)
    % u{1} = ux = velocity in x direction
    % u{2} = uy = velocity in y direction

    %% Input processing
    % if nargin < 5
    %   uMode = 'min';
    % end
    % ONLY DOING BRAT 
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

        % inside BRAT
        if current_data_v1 <= -schemeData.v1_threshold
            uOpt = analytical_optimal_control(deriv, x);
        % outside BRAT
        elseif current_data_v1 > 0
            uOpt = eval_u(schemeData.grid_with_time, schemeData.opt_v1_control, [x', t]);
            uOpt = num2cell(uOpt);
        
        % on the boundary
        else
            % compute max margin
            yalmip('clear');
            u_v1_eq0 = sdpvar(2, 1, 'full');
            current_deriv_v1 = eval_u(schemeData.grid_with_time, schemeData.deriv_v1, [x', t]);
            Constraints = [control_bound_func_handle(u_v1_eq0)];
            Objective = (-current_deriv_v1(3) + current_deriv_v1(1:2)' * f(x, u_v1_eq0));
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            sol = optimize(Constraints, Objective, Options);
            if sol.problem == 0
                best_constraint = value(Objective);
                best_constraint = max(best_constraint, 0);
            else
                error('In max margin computation, the following error occurred: %s', sol.info);
            end
            
            % compute opt ctrl on brt boundary
            yalmip('clear');
            current_deriv_v2 = cell2mat(deriv); 
            u_v1_eq0 = sdpvar(2, 1);
            Constraints = [control_bound_func_handle(u_v1_eq0), ...
                -current_deriv_v1(3) + current_deriv_v1(1:2)' * f(x, u_v1_eq0) <=  best_constraint];
            Objective = (L(x, u_v1_eq0) + current_deriv_v2' * f(x, u_v1_eq0));
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            sol = optimize(Constraints, Objective, Options);
            if sol.problem == 0
                uOpt = num2cell(value(u_v1_eq0));
            else
                error('In opt ctrl computation, the following error occurred: %s', sol.info);
            end
        end
        
    else % When x is a grid 
        if size(deriv{1}, 1) == 1 && size(deriv{1}, 2) == 1 % account for calls from genericPartial
            deriv{1} = ones(size(x{1})) .* deriv{1};
            deriv{2} = ones(size(x{2})) .* deriv{2};
        end
        
        [~, v1_idx] = min(abs(t - schemeData.tau));
 
        %%%%%%%%%%%%%%%% For v1 > 0 (outside of BRAT) %%%%%%%%%%%%%%%%
        u_opt{1} = schemeData.opt_v1_control{1}(:,:,v1_idx) .* schemeData.v1_g0_mask(:,:,v1_idx);
        u_opt{2} = schemeData.opt_v1_control{2}(:,:,v1_idx) .* schemeData.v1_g0_mask(:,:,v1_idx);

        %%%%%%%%%%%%%%%% For v1 < -v1_threshold (inside of BRAT) %%%%%%%%%%%%%%%%
        if schemeData.use_analytical_optimal_control
            u_in_brat = analytical_optimal_control(deriv, schemeData.grid.xs);
            u_opt{1} = u_opt{1} + u_in_brat{1} .* schemeData.v1_l0_mask(:,:,v1_idx);
            u_opt{2} = u_opt{2} + u_in_brat{2} .* schemeData.v1_l0_mask(:,:,v1_idx);
        else
            disp('here')
            yalmip('clear');
            num_states_v1_g0 = length(schemeData.v1_g0_subscript_x{v1_idx});
            u_v1_g0 = sdpvar(num_states_v1_g0, 2);
            Constraints = [];
            Objective = 0;
            for i = 1:num_states_v1_g0
                xIdx = schemeData.v1_g0_subscript_x{v1_idx}(i);
                yIdx = schemeData.v1_g0_subscript_y{v1_idx}(i);
                state = [schemeData.grid.xs{1}(xIdx, yIdx); schemeData.grid.xs{2}(xIdx, yIdx)];
    
                deriv_at_point = [deriv{1}(xIdx, yIdx); deriv{2}(xIdx, yIdx)];
                Constraints = [Constraints, ...
                    control_bound_func_handle(u_v1_g0(i,:))];
                Objective = Objective + L(state, u_v1_g0(i,:)) + deriv_at_point' * f(state, u_v1_g0(i,:));
            end
            Options = sdpsettings('verbose', 0, 'solver', 'gurobi');
            sol = optimize(Constraints, Objective, Options);
            if sol.problem == 0
                u_v1_g0 = value(u_v1_g0);
            else
                error('The following error occurred: %s', sol.info);
            end
            for i = 1:num_states_v1_g0
                xIdx = schemeData.v1_g0_subscript_x{v1_idx}(i);
                yIdx = schemeData.v1_g0_subscript_y{v1_idx}(i);
                u_opt{1}(xIdx, yIdx) = u_v1_g0(i, 1);
                u_opt{2}(xIdx, yIdx) = u_v1_g0(i, 2);
            end
        end

        %%%%%%%%%%%%%%%% For v1 = 0 %%%%%%%%%%%%%%%%
        yalmip('clear');
        num_states_v1_eq0 = length(schemeData.v1_eq0_subscript_x{v1_idx});
        u_v1_eq0 = sdpvar(num_states_v1_eq0, 2, 'full');
        Constraints = [];
        Objective = 0;
        for i = 1:num_states_v1_eq0
            xIdx = schemeData.v1_eq0_subscript_x{v1_idx}(i);
            yIdx = schemeData.v1_eq0_subscript_y{v1_idx}(i);
            state = [schemeData.grid.xs{1}(xIdx, yIdx); schemeData.grid.xs{2}(xIdx, yIdx)];
            current_deriv_v1 = [schemeData.deriv_v1{1}(xIdx, yIdx, v1_idx); ...
                schemeData.deriv_v1{2}(xIdx, yIdx, v1_idx); ...
                schemeData.deriv_v1{3}(xIdx, yIdx, v1_idx)];
            deriv_at_point = [deriv{1}(xIdx, yIdx); deriv{2}(xIdx, yIdx)];
            Constraints = [Constraints, control_bound_func_handle(u_v1_eq0(i,:))];
            Constraints = [Constraints, ...
                -current_deriv_v1(3) + current_deriv_v1(1:2)' * f(state, u_v1_eq0(i,:)) <=  schemeData.best_constraints_eq0{v1_idx}(i) + 1e-4];
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
            xIdx = schemeData.v1_eq0_subscript_x{v1_idx}(i);
            yIdx = schemeData.v1_eq0_subscript_y{v1_idx}(i);
            u_opt{1}(xIdx, yIdx) = u_v1_eq0(i, 1);
            u_opt{2}(xIdx, yIdx) = u_v1_eq0(i, 2);
        end
        uOpt{1} = u_opt{1};
        uOpt{2} = u_opt{2};
    end

    % check if opt ctrl is unbounded
    if any(isnan(uOpt{1}))
        error('opt ctrl is unbounded')
    end
    if any(isnan(uOpt{2}))
        error('opt ctrl is unbounded')
    end
end

function u_opt = analytical_optimal_control(deriv, x)
    u_opt = cell(2, 1);
    deriv_x1_x2_norm = (deriv{1} .^2 + deriv{2} .^ 2) .^ 0.5; 
    deriv_x1_x2_norm = deriv_x1_x2_norm .* (deriv_x1_x2_norm > 0) + 1 * (deriv_x1_x2_norm <= 1e-8); % avoid division by zero
    

    deriv_x1_normalized = deriv{1} ./ deriv_x1_x2_norm; 
    deriv_x2_normalized = deriv{2} ./ deriv_x1_x2_norm;
    u_opt{1} = -deriv_x1_normalized .* (deriv_x1_x2_norm >= 1) + 0 .* (deriv_x1_x2_norm < 1);
    u_opt{2} = -deriv_x2_normalized .* (deriv_x1_x2_norm >= 1) + 0 .* (deriv_x1_x2_norm < 1); 
end

function cost = L(x, u)
    if ~iscell(x)
        x = num2cell(x);
    end
    if ~iscell(u)
        u = num2cell(u);
    end
    cost = sqrt(u{1}.^2 + u{2}.^2);
end