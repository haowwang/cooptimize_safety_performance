function [data] = computeStateConsValueFunc(grid, dyn_sys, obstacles, targets, final_cost_func_handle, tau, v1_mode, extra_args)
% Important note regarding dynSys: the optCtrl must return the ctrl that optimizes the Hamiltonian

    if strcmp(v1_mode, 'brt_avoid') == 1
        final_cost = final_cost_func_handle(grid.xs);
        data0 = final_cost - grid.xs{end}; 
        HJIextraArgs.obstacleFunction = obstacles; % g(x)
        compMethod = 'none'; % the state constrained oc HJB-VI is almost identical 
            % to the VI for the backward reach-avoid set. hence compMethod = 'none'
    elseif strcmp(v1_mode, 'brat') == 1
        data0 = max(final_cost_func_handle(grid.xs) - grid.xs{3}, targets);
        HJIextraArgs.obstacleFunction = obstacles; 
        HJIextraArgs.targetFunction = max(final_cost_func_handle(grid.xs) - grid.xs{3}, targets); 
        compMethod = 'minVWithL'; 
    elseif strcmp(v1_mode, 'bras') == 1
        data0 = max(final_cost_func_handle(grid.xs) - grid.xs{3}, targets);
        HJIextraArgs.obstacleFunction = obstacles; 
        compMethod = 'none'; 
    else
        error('Unknown v1 mode');
    end


    %% Pack the problem parameters for dynamic programming
    schemeData.dynSys = dyn_sys; 
    schemeData.grid = grid; % Grid over which the value function is computed
    if isfield(extra_args, 'custom_ham')
        disp('Using custom Hamiltonian');
        schemeData.hamFunc = extra_args.custom_ham;
    else
        disp('Using generic Hamiltonian');
        schemeData.hamFunc = @genericStateConsHam;
    end

    schemeData.uMode = 'min'; % Control is trying to minimize the cost function
    schemeData.dMode = 'max'; % Disturbance is trying to maximize the cost function
    schemeData.tMode = 'backward';
    schemeData.accuracy = 'veryHigh';
    schemeData.use_analytical_opt_ctrl = extra_args.use_analytical_opt_ctrl;

    HJIextraArgs.visualize.initialValueFunction = 0;
    HJIextraArgs.visualize.valueFunction = 1;
    HJIextraArgs.visualize.figNum = 1; %set figure number
    HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
    HJIextraArgs.visualize.xTitle = 'x position';
    HJIextraArgs.visualize.yTitle = 'y position';
    HJIextraArgs.visualize.zTitle = 'Value Function';
    HJIextraArgs.visualize.viewGrid = true;
    HJIextraArgs.visualize.plotData.plotDims = [1 1 0]; % plot x, y, slice
    HJIextraArgs.visualize.plotData.projpt = [0]; % slice at theta angle
    
    tic
    % solve for the augmented value function. 
    [data, tau, ~] = HJIPDE_solve(data0, tau, schemeData, compMethod, HJIextraArgs);
    toc 





end