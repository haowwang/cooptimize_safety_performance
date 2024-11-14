function uOpt = optCtrl(obj, ~, x, deriv, schemeData)

  % % Important note: this is written for running cost = l2 norm of ctrl
  uMode = schemeData.uMode;
  if ~iscell(deriv)
    deriv = num2cell(deriv);
  end
  uOpt = cell(obj.nu,1);

  deriv_x1_x2_norm = (deriv{1} .^ 2 + deriv{2} .^ 2) .^ 0.5;

  % for numerical stability
  deriv_x1_x2_norm = deriv_x1_x2_norm .* (deriv_x1_x2_norm > 0) + 1 * (deriv_x1_x2_norm < 1e-8);

  % vector opposite of the gradient of the vfunc (in x1 and x2 direction)
  opt_u_dir_x1 = - deriv{1} ./ deriv_x1_x2_norm;
  opt_u_dir_x2 = - deriv{2} ./ deriv_x1_x2_norm; 

  neg_deriv_x1_x2_norm_minus_deriv_x3 = -deriv_x1_x2_norm - deriv{3}; 

  if strcmp(uMode, 'min')
    uOpt{1} = (neg_deriv_x1_x2_norm_minus_deriv_x3 < 0) .* opt_u_dir_x1 + (neg_deriv_x1_x2_norm_minus_deriv_x3 >= 0) .* 0;
    uOpt{2} = (neg_deriv_x1_x2_norm_minus_deriv_x3 < 0) .* opt_u_dir_x2 + (neg_deriv_x1_x2_norm_minus_deriv_x3 >= 0) .* 0; 
  else
    error('uMode should be min.');
  end

  % check if opt ctrl is unbounded
  if any(isnan(uOpt{1}))
    keyboard
  end
  if any(isnan(uOpt{2}))
      keyboard
  end

end