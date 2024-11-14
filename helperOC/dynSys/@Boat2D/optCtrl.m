function uOpt = optCtrl(obj, ~, x, deriv, schemeData)
% u{1} = ux = velocity in x direction
% u{2} = uy = velocity in y direction

%% Input processing
% if nargin < 5
%   uMode = 'min';
% end
uMode = schemeData.uMode;
if ~iscell(deriv)
  deriv = num2cell(deriv);
end
uOpt = cell(obj.nu, 1);

deriv_x1_x2_norm = (deriv{1} .^2 + deriv{2} .^ 2) .^ 0.5; 
deriv_x1_x2_norm = deriv_x1_x2_norm .* (deriv_x1_x2_norm > 0) + 1 * (deriv_x1_x2_norm <= 1e-8); % avoid division by zero

deriv_x1_normalized = deriv{1} ./ deriv_x1_x2_norm; 
deriv_x2_normalized = deriv{2} ./ deriv_x1_x2_norm;

%% Optimal control
if strcmp(uMode, 'max')
  uOpt{1} = deriv_x1_normalized * obj.vMax;
  uOpt{2} = deriv_x2_normalized * obj.vMax;
elseif strcmp(uMode, 'min')
  uOpt{1} = -deriv_x1_normalized * obj.vMax;
  uOpt{2} = -deriv_x2_normalized * obj.vMax;
else
  error('Unknown uMode!')
end

% check if opt ctrl is unbounded
if any(isnan(uOpt{1}))
    keyboard
end
if any(isnan(uOpt{2}))
    keyboard
end


end