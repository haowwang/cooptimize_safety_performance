function uOpt = optCtrl(obj, ~, x, deriv, schemeData)
  % Important note: this is written for ctrl independent running cost
  uMode = schemeData.uMode;
  uOpt = cell(obj.nu,1);
  if ~iscell(deriv)
    deriv = num2cell(deriv);
  end
  if ~iscell(x)
    x = num2cell(x);
  end

  if schemeData.use_analytical_opt_ctrl
    deriv_x1_x2_norm = (deriv{1} .^ 2 + deriv{2} .^ 2) .^ 0.5;
    % for numerical stability
    deriv_x1_x2_norm = deriv_x1_x2_norm .* (deriv_x1_x2_norm > 0) + 1 * (deriv_x1_x2_norm < 1e-8);
    if strcmp(uMode, 'min')
      % vector opposite of the gradient of the vfunc (in x1 and x2 direction)
        uOpt{1} = - obj.vMax .* deriv{1} ./ deriv_x1_x2_norm;
        uOpt{2} = - obj.vMax .* deriv{2} ./ deriv_x1_x2_norm;
    else
      error('uMode should be min.');
    end

  else
    if size(x{1}, 1) == 1 && size(x{1}, 2) == 1 && size(x{1}, 3) == 1
      % c = [deriv{1}; deriv{2}];
      model.A =sparse(1, 2);
      model.rhs = 0;
      model.quadcon(1).Qc = sparse([1 0; 0 1]);
      model.quadcon(1).q = [0 0];
      model.quadcon(1).rhs = 1;
      model.obj = [deriv{1}; deriv{2}];
      model.lb = [-1; -1];
      params.OutputFlag = 0;
      result = gurobi(model, params);
      if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
          u_opt = result.x;
      else
          error('The following error occurred: %s', result.status);
      end
      uOpt{1} = u_opt(1);
      uOpt{2} = u_opt(2);
      % u_opt = sdpvar(2, 1);
      % constraints = [u_opt(1)^2 + u_opt(2)^2 <= 1];
      % objective = deriv{1} * u_opt(1) + deriv{2} * u_opt(2);
      % options = sdpsettings('verbose', 0, 'solver', 'gurobi');
      % sol = optimize(constraints, objective, options);
      % if sol.problem == 0
      %   uOpt{1} = value(u_opt(1));
      %   uOpt{2} = value(u_opt(2));
      % else
      %   error('The following error occurred: %s', sol.info);
      % end

    else 
      if size(deriv{1}, 1) == 1 && size(deriv{1}, 2) == 1 && size(deriv{1}, 3) == 1 % account for calls from genericPartial
        deriv{1} = ones(size(x{1})) .* deriv{1};
        deriv{2} = ones(size(x{2})) .* deriv{2};
        deriv{3} = ones(size(x{3})) .* deriv{3};
      end

      state_dim = schemeData.dynSys.nx - 1;
      ctrl_dim = schemeData.dynSys.nu; 
      num_grid_points = 1.;
      state_stacked = []; % of dimension (grid_resol, ... , state_dim)
      for i = 1:1:schemeData.grid.dim
          num_grid_points = num_grid_points * schemeData.grid.N(i);
      end

      start_time = tic;
      model.A = sparse(num_grid_points, ctrl_dim*num_grid_points);
      model.rhs = zeros(num_grid_points, 1);

      c1 = deriv{1}(:);
      c2 = deriv{2}(:);
      c = [c1; c2];
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

      model.obj = c;
      model.lb = -ones(ctrl_dim * num_grid_points, 1);
      params.OutputFlag = 0;
      % model.sense = '=';
      result = gurobi(model, params);
      if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'SUBOPTIMAL')
        uOpt_stacked = result.x;
      else
          error('The following error occurred: %s', result.status);
      end
      uOpt{1} = reshape(uOpt_stacked(1:num_grid_points), [schemeData.grid.N(1), schemeData.grid.N(2), schemeData.grid.N(3)]);
      uOpt{2} = reshape(uOpt_stacked(num_grid_points + 1:2*num_grid_points), [schemeData.grid.N(1), schemeData.grid.N(2), schemeData.grid.N(3)]);
      end_time = toc(start_time);
      % fprintf('Time taken for optimization: %f\n', end_time);
    end
  end

if any(isnan(uOpt{1}))
  keyboard
end
if any(isnan(uOpt{2}))
  keyboard
end


end