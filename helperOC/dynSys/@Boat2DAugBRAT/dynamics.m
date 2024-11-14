function dx = dynamics(obj, ~, x, u, d, schemeData)
% Dynamics:
      %    \dot{x}_1 = v * cos(u) + c - a *  x_2^2
      %    \dot{x}_2 = v * sin(u) 
      %    \dot{x}_3 = l(t,x,u) % running cost evolution
      %         v \in [0, vRange(2)]
      %         u \in [0, 2*pi]


if ~iscell(x)
  x = num2cell(x);
  u = num2cell(u);
  d = num2cell(d);
end

dx = cell(length(obj.dims), 1);

for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
end


% if iscell(x)
%   
% 
%   for i = 1:length(obj.dims)
%     dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
%   end
% else
%   dx = zeros(obj.nx, 1);
% 
%   dx(1) = obj.speed * cos(x(3)) + d(1);
%   dx(2) = obj.speed * sin(x(3)) + d(2);
%   dx(3) = u + d(3);
% end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = u{1} + obj.c - obj.a * x{2} .^ 2;
  case 2
    dx = u{2};
  case 3 
    dx = -obj.running_cost_func_handle_cell(x,u); % running cost as implemented in state constrained HJB paper -> l
  otherwise
    error('Only dimension 1-3 are defined for dynamics of Boat2D!')
end
end