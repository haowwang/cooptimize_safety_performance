classdef Boat2DCtrlConsBRAT < DynSys % regular Boat2D without the virtual state
  properties
    % speed bound [0, some postive real]
    vMax

    % Dimensions that are active
    dims

    c % constant drift

    a % coefficient for x_1 state dependent drift

    dyn_func_handle % dynamics function handle

    control_bound_func_handle % control bound function handle

    running_cost_func_handle

  end
  
  methods
      function obj = Boat2DCtrlConsBRAT(x, vMax, c, a, running_cost_func_handle)
      % obj = DubinsCar(x, wMax, speed, dMax, dims)
      %     Boat2D class
      %
      % Dynamics:
      %    \dot{x}_1 = ux + c - a *  x_2^2
      %    \dot{x}_2 = ux
      %         ux^2 + uy^2 <= vMax^2
      %
      % Inputs:
      %   x      - state: [xpos; ypos]
      %   thetaMin   - minimum angle
      %   thetaMax   - maximum angle
      %
      % Output:
      %   obj       - a Boat2D object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      if nargin < 2
        vMax = 1;
      end
      
      if nargin < 3
        c = 0; % no constant drift
      end
      
      if nargin < 4
        a = 0; % no y dependent drift
      end
      
      % Basic vehicle properties
      obj.pdim = 1:2; % Position dimensions
      %obj.hdim = find(dims == 3);   % Heading dimensions
      obj.nx = 2;
      obj.nu = 2;

      obj.vMax = vMax; 
      obj.c = c; 
      obj.a = a; 
      
      obj.x = x;
      obj.xhist = obj.x;

      obj.dims = 1:2;
      obj.dyn_func_handle = @(x, u) [u(1) + obj.c - obj.a * (x(2)^2); u(2)];
      obj.control_bound_func_handle = @(u) norm(u, 2) <= obj.vMax;
      obj.running_cost_func_handle = running_cost_func_handle;
    end
    
  end % end methods
end % end classdef
