function hamValue = genericCoopHam(t, data, deriv, schemeData)

    u_opt = schemeData.dynSys.optCtrl(t, schemeData.grid.xs, deriv, schemeData);
    L = schemeData.dynSys.running_cost_func_handle;

    hamValue = deriv{1} .* 0;
    state_dot = schemeData.dynSys.dynamics([], schemeData.grid.xs, u_opt, {0, 0}, schemeData);
    for j = 1:schemeData.dynSys.nx
        hamValue = hamValue + deriv{j} .* state_dot{j};
    end
    hamValue = hamValue + L(schemeData.grid.xs, u_opt);


    if strcmp(schemeData.tMode, 'backward')
        hamValue = -hamValue;
    else
        error('tMode must be ''backward''!')
    end
end