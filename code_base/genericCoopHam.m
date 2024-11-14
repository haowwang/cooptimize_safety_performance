function hamValue = genericCoopHam(t, data, deriv, schemeData)

    u_opt = schemeData.dynSys.optCtrl(t, schemeData.grid.xs, deriv, schemeData);
    L = schemeData.dynSys.running_cost_func_handle;

    hamValue = deriv{1} .* 0;
    state_dot = schemeData.dynSys.dynamics([], schemeData.grid.xs, u_opt, {0, 0}, schemeData);
    for j = 1:schemeData.dynSys.nx
        hamValue = hamValue + deriv{j} .* state_dot{j};
    end
    hamValue = hamValue + L(schemeData.grid.xs, u_opt);

    
    
    % close all;
    % [~, v1_idx] = min(abs(t - schemeData.tau));
    % 
    % figure; axis equal; hold on; 
    % quiver(schemeData.grid.xs{1}, schemeData.grid.xs{2}, u_opt{1}, u_opt{2});
    % contour(schemeData.grid.xs{1}, schemeData.grid.xs{2}, schemeData.v1_data(:,:,v1_idx),...
    % [0,0], 'LineWidth',2, 'EdgeColor','r');
    % 
    % figure; hold on; axis equal; 
    % deriv_norm = (deriv{1}.^2 + deriv{2}.^2).^2; 
    % contourf(schemeData.grid.xs{1}, schemeData.grid.xs{2}, deriv_norm); 
    % colorbar; 
    % % contour(schemeData.grid.xs{1}, schemeData.grid.xs{2}, schemeData.v1_data(:,:,v1_idx),...
    % % [0,0], 'LineWidth',2, 'EdgeColor','r');
    % viscircles([1.5, 0], [0.25], 'Color','g');
    % title('deriv norm'); 
    % 
    % figure; hold on; axis equal; 
    % title('deriv v2');
    % quiver(schemeData.grid.xs{1}, schemeData.grid.xs{2}, deriv{1}, deriv{2});
    % viscircles([1.5, 0], [0.25], 'Color','g');
    % % 
    % figure; hold on; axis equal; 
    % title('ham value'); 
    % contourf(schemeData.grid.xs{1}, schemeData.grid.xs{2}, hamValue); 
    % colorbar; 
    % viscircles([1.5, 0], [0.25], 'Color','g');
    % % 
    % figure; axis equal; hold on; 
    % title('value v2'); 
    % contourf(schemeData.grid.xs{1}, schemeData.grid.xs{2}, data); 
    % colorbar; 
    % viscircles([1.5, 0], [0.25], 'Color','g');
    % 
    % figure; axis equal;hold on; 
    % quiver(schemeData.grid.xs{1}, schemeData.grid.xs{2}, deriv{1}, deriv{2});
    % viscircles([1.5, 0], [0.25], 'Color','g');
    % title('deriv v2');
    % 
    % figure; axis equal;hold on; 
    % quiver(schemeData.grid.xs{1}, schemeData.grid.xs{2}, u_opt{1}, u_opt{2});
    % viscircles([1.5, 0], [0.25], 'Color','g');
    % title('u opt');


    if strcmp(schemeData.tMode, 'backward')
        hamValue = -hamValue;
    else
        error('tMode must be ''backward''!')
    end
end