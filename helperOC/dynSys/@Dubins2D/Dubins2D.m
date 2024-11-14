classdef Dubins2D < DynSys
  properties
    % Angle bounds
    wRange
    
    speed % Constant speed
    
    % Disturbance
    dRange
    
    % Dimensions that are active
    dims

    % \dot{x} = f(x) + g(x)u
    dyn_func_handle % Dynamics function

    control_bound_func_handle % Control bounds
  end
  
  methods
      function obj = Dubins2D(x, wRange, dims, dRange, dyn_func_handle, ctrl_constraint)
      % obj = DubinsCar(x, wMax, speed, dMax, dims)
      %     Dubins Car class
      %
      % Dynamics:
      %    \dot{x}_1 = v * cos(x_3) + d1
      %    \dot{x}_2 = v * sin(x_3) + d2
      %    \dot{x}_3 = u
      %         u \in [-wMax, wMax]
      %         d \in [-dMax, dMax]
      %
      % Inputs:
      %   x      - state: [xpos; ypos]
      %   thetaMin   - minimum angle
      %   thetaMax   - maximum angle
      %   v - speed
      %   dMax   - disturbance bounds
      %
      % Output:
      %   obj       - a DubinsCar2D object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      if nargin < 2
        wRange = [-1 1];
      end
      
      
      if nargin < 4
        dRange = {[0; 0];[0; 0]};
      end
      
      if nargin < 3
        dims = 1:2;
      end
      
      if numel(wRange) <2
          wRange = [-wRange; wRange];
      end
      % 
      % if ~iscell(dRange)
      %     dRange = {-dRange;dRange};
      % end
      
      % Basic vehicle properties
      obj.pdim = [find(dims == 1) find(dims == 2)]; % Position dimensions
      %obj.hdim = find(dims == 3);   % Heading dimensions
      obj.nx = length(dims);
      obj.nu = 2;
      obj.nd = 2;
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.wRange = wRange;
      %obj.thetaMax = thetaMax;
      obj.dRange = dRange;
      obj.dims = dims;

      obj.dyn_func_handle = dyn_func_handle;
      obj.control_bound_func_handle = ctrl_constraint;
    end
    
  end % end methods
end % end classdef
