function dx = dynamics(obj, ~, x, u, d, ~)
% Dynamics of the Dubins Car
%    \dot{x}_1 = vx + d1
%    \dot{x}_2 = vy + d2
%   Control: u = [vx;vy];

if iscell(x)
  dx = cell(length(obj.dims), 1);
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
  end
else
  dx = zeros(obj.nx, 1);
  dx(1) = u(1) + d(1);
  dx(2) = u(2) + d(2);
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = u{1} + d{1};
  case 2
    dx = u{2} + d{2};
  otherwise
    error('Only dimension 1-2 are defined for dynamics of Dubins2D!')
end
end