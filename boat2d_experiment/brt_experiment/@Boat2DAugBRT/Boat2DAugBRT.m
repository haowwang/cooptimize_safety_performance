classdef Boat2DAugBRT < DynSys
  properties
    dims

    vMax % control normal bound

    c % constant drift

    a % coefficient for x_1 state dependent drift

    ctrl_bound_func_handle_arr % function handle for control bound. used in ham calculation

    dyn_func_handle_arr % function handle for system dynamics (excluding the auxiliary variable). used in ham calculation

    running_cost_func_handle_cell % function handle for running cost. only for cell array

    running_cost_func_handle_arr % function handle for running cost. only for array
  end
  
  methods
      function obj = Boat2DAugBRT(x, vMax, c, a, running_cost_func_handle_cell, running_cost_func_handle_arr)
      % obj = DubinsCar(x, wMax, speed, dMax, dims)
      %     Boat2D class
      %
      % Dynamics:
      %    \dot{x}_1 = u_1 + c - a *  x_2^2
      %    \dot{x}_2 = u_2
      %    \dot{x}_3 = ||[u_1, u_2]|| % running cost as implemented in the state
      %    constrained HJB PDE paper
      %         ||[u_1, u_2]|| <= 1
      %
      % Inputs:
      %   x      - state: [xpos; ypos]
      %
      % Output:
      %   obj       - a Boat2DAug object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      if nargin < 2
        vMax = 1.; 
      end
      
      if nargin < 3
        c = 0; % no constant drift
      end
      
      if nargin < 4
        a = 0; % no y dependent drift
      end
      
      % Basic vehicle properties
      obj.dims = 1:3;
      obj.pdim = 1:2; % Position dimensions
      obj.nx = 3;
      obj.nu = 2;
      
      obj.c = c; 
      obj.a = a; 
      obj.vMax = vMax; 

      obj.x = x;
      obj.xhist = obj.x;

      obj.dyn_func_handle_arr = @(state,ctrl) [ctrl(1) + obj.c - obj.a * state(2)^2; ctrl(2)]; 
      obj.ctrl_bound_func_handle_arr = @(ctrl_var) norm(ctrl_var, 2) <= obj.vMax;
      obj.running_cost_func_handle_cell = running_cost_func_handle_cell;
      obj.running_cost_func_handle_arr = running_cost_func_handle_arr;

    end

  end % end methods

end % end classdef
