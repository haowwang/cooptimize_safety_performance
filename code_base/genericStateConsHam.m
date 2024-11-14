function hamValue = genericStateConsHam(t, data, deriv, schemeData)
    
    state_dim = schemeData.dynSys.nx - 1;
    uOpt = schemeData.dynSys.optCtrl(t, schemeData.grid.xs, deriv, schemeData);
    state_dot = schemeData.dynSys.dynamics(t, schemeData.grid.xs, uOpt, schemeData); 

    hamValue = 0.;
    for i = 1:1:state_dim
      hamValue = hamValue + (deriv{i} .* state_dot{i});
    end

    hamValue = hamValue + (-deriv{end} .* schemeData.dynSys.running_cost_func_handle_cell(schemeData.grid.xs, uOpt));
  
    
    if strcmp(schemeData.tMode, 'backward')
      hamValue = -hamValue;
    else
      error('tMode must be ''backward''!')
    end

end