function dx = dynamics(obj, ~, x, u, d, schemeData)
% Dynamics:
      %    \dot{x}_1 = v * cos(u) + c - a *  x_2^2
      %    \dot{x}_2 = v * sin(u) 
      %         v \in [0, vRange(2)]
      %         u \in [0, 2*pi]


% if nargin < 5
%   d = [0; 0; 0];
% end

% if ~iscell(x)
%   x = num2cell(x);
%   u = num2cell(u);
%   d = num2cell(d);
% end

% dx = cell(length(obj.dims), 1);

% for i = 1:length(obj.dims)
%     dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
% end


if iscell(x)
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
  end
else
  dx = zeros(obj.nx, 1);
  dx(1) = u(1) + obj.c - obj.a * (x(2)^ 2);
  dx(2) = u(2);
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = u{1} + obj.c - obj.a .* (x{2} .^ 2);
  case 2
    dx = u{2};
  otherwise
    error('Only dimension 1-2 are defined for dynamics of Boat2D!')
end
end