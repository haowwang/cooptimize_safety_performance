function uOpt = optCtrl(obj, ~, x, deriv, schemeData)
  % uOpt = optCtrl(obj, t, y, deriv, uMode)
  % assuming control bounds on u1 and u2 are intervals (i.e. u = [u1, u2] is
  % bounded by l-2 norm)
  %% Input processing
  % if nargin < 5
  %   uMode = 'min';
  % end
  uMode = schemeData.uMode;
  if ~iscell(deriv)
    deriv = num2cell(deriv);
  end
  uOpt = cell(obj.nu,1);
  %% Optimal control

  deriv_l2_norm = (deriv{1} .^ 2 + deriv{2} .^ 2) .^ 0.5;
 
  % for numerical stability (the origin always have gradient of 0)
  deriv_l2_norm = deriv_l2_norm .* (deriv_l2_norm > 0) + 1 * (deriv_l2_norm < 1e-8);
  a = sum(deriv_l2_norm < 0.001, 'all');
  % if a > 0
  %     disp(a)
  %     error('deriv_l2_norm is too small')
  % end
  if strcmp(uMode, 'max')
    uOpt{1} = (deriv{1} ./ deriv_l2_norm) * obj.wRange(2);
    uOpt{2} = (deriv{2} ./ deriv_l2_norm) * obj.wRange(2);
    if any(isnan(uOpt{1}))
        error('opt ctrl is unbounded')  
    end
    if any(isnan(uOpt{2}))
        error('opt ctrl is unbounded')
    end
  elseif strcmp(uMode, 'min')
    uOpt{1} = -(deriv{1} ./ deriv_l2_norm) * obj.wRange(2);
    uOpt{2} = -(deriv{2} ./ deriv_l2_norm) * obj.wRange(2);
  end
end