function dOpt = optDstb(obj, ~, x, deriv, uOpt, schemeData)
% dOpt = optCtrl(obj, t, y, deriv, dMode)
%     Dynamics of the DubinsCar
%         \dot{x}_1 = v * cos(x_3) + d_1
%         \dot{x}_2 = v * sin(x_3) + d_2
% important note: assuming disturbance is bounded by some l-infinity norm
%% Input processing
% if nargin < 5
%   dMode = 'max';
% end
dMode = schemeData.dMode;
if ~iscell(deriv)
  deriv = num2cell(deriv);
end
dOpt = cell(obj.nd, 1);

abs_deriv_1 = abs(deriv{1}); 
abs_deriv_2 = abs(deriv{2});
max_abs_deriv = max(abs_deriv_1, abs_deriv_2); % take element wise max; normalization factor for l-infinity norm
if strcmp(dMode, 'max')
  dOpt{1} = (deriv{1} ./ max_abs_deriv) * obj.dRange{1}(2);
  dOpt{2} = (deriv{2} ./ max_abs_deriv) * obj.dRange{1}(2);
elseif strcmp(dMode, 'min')
  dOpt{1} = -(deriv{1} ./ max_abs_deriv) * obj.dRange{1}(2);
  dOpt{2} = -(deriv{2} ./ max_abs_deriv) * obj.dRange{1}(2);
else
  error('Unknown uMode!')
end
dOpt{1} = deriv{1} .* 0;
dOpt{2} = deriv{2} .* 0;
end