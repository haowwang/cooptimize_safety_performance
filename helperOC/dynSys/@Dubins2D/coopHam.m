function hamValue = coopHam(t, data, deriv, schemeData)
checkStructureFields(schemeData, 'grid', 'runningCost', 'states_on_brt_boundary_subscript_x', ...
     'states_on_brt_boundary_subscript_y', 'deriv_safety_states_on_brt_boundary', 'vMax');

% default goal seeking optimal control
deriv_l2_norm = (deriv{1} .^ 2 + deriv{2} .^ 2) .^ 0.5; 
deriv_l2_norm = deriv_l2_norm .* (deriv_l2_norm > 0) + 1 * (deriv_l2_norm == 0); % avoid division by zero
if sum(deriv_l2_norm == 0., 'all') > 0
    keyboard
end

normalized_deriv{1} = deriv{1} ./ deriv_l2_norm;
normalized_deriv{2} = deriv{2} ./ deriv_l2_norm;

opt_ctrl{1} = -normalized_deriv{1} * schemeData.vMax;
opt_ctrl{2} = -normalized_deriv{2} * schemeData.vMax;

num_states_on_brt_boundary = size(schemeData.states_on_brt_boundary_subscript_x, 1); 

yalmip('clear');
ctrl_vars = sdpvar(num_states_on_brt_boundary, 2); % optimal control at states on the brt boundary
constraints = [];
objective = 0.; 

% loop through boundary point indices and add safety half space constraint
for i = 1:1:num_states_on_brt_boundary
        constraints = [constraints, norm(ctrl_vars(i,:)) <= 1 * schemeData.vMax];
        % constraints = [ctrl_vars(1,1) <= schemeData.vMax, ctrl_vars(1,2) <= schemeData.vMax];
        constraints = [constraints, schemeData.deriv_safety_states_on_brt_boundary{1}(i) * ctrl_vars(i,1) + ...
            schemeData.deriv_safety_states_on_brt_boundary{2}(i) * ctrl_vars(i,2) >= 0];
        perf_deriv_x_state_on_brt_boundary = deriv{1}(schemeData.states_on_brt_boundary_linear_idx(i));
        perf_deriv_y_state_on_brt_boundary = deriv{2}(schemeData.states_on_brt_boundary_linear_idx(i));
        objective = objective + perf_deriv_x_state_on_brt_boundary * ctrl_vars(i,1) + perf_deriv_y_state_on_brt_boundary ...
            * ctrl_vars(i,2);
    
end

options = sdpsettings('verbose', 0, 'solver', 'gurobi');
sol = optimize(constraints, objective, options);
if sol.problem > 0 
    error('In the loop Hamiltonian optimization failed');
end
x_sol = value(ctrl_vars);

for i = 1:1:num_states_on_brt_boundary
    opt_ctrl{1}(schemeData.states_on_brt_boundary_linear_idx(i)) = x_sol(i, 1); 
    opt_ctrl{2}(schemeData.states_on_brt_boundary_linear_idx(i)) = x_sol(i, 2); 
    if x_sol(i, 1) < 1e-6 && x_sol(i, 2) < 1e-6 % optimizer can't find the orthogonal solution. manually set the opt ctrl
        opt_ctrl{1}(schemeData.states_on_brt_boundary_linear_idx(i)) = schemeData.deriv_safety_states_on_brt_boundary{2}(i);
        opt_ctrl{2}(schemeData.states_on_brt_boundary_linear_idx(i)) = - schemeData.deriv_safety_states_on_brt_boundary{1}(i);
    end
end

%% Compute Hamiltonian
% Running cost - constant term of the Hamiltonian 
hamRunning = schemeData.runningCost;

% Control component - min_u p1*v_x + p2*v_y
hamCtrl = opt_ctrl{1} .* deriv{1} + opt_ctrl{2} .* deriv{2};

% Disturbance component - max_d p1*d_x + p2*d_y
% hamDstb = dMax * (abs(deriv{1}) + abs(deriv{2}));

hamValue = hamRunning + hamCtrl;

%% Backward or forward reachable set
if strcmp(schemeData.tMode, 'backward')
  hamValue = -hamValue;
else
  error('tMode must be ''backward''!')
end


end