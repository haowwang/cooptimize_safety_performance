function [data, deriv_v1, opt_v1_ctrl] = computeV1(grid, dyn_sys, obstacles, targets, tau, v1_mode)
% v1_mode -> brt_reach, brt_avoid, brat, bras
    if strcmp(v1_mode, 'brt_reach')
        uMode = 'min';
        dMode = 'max';
        compMode = 'minVOverTime';
        data0 = targets;
    elseif strcmp(v1_mode, 'brt_avoid')
        uMode = 'max';
        dMode = 'min';
        compMode = 'minVOverTime';
        data0 = obstacles;
    elseif strcmp(v1_mode, 'brat')
        uMode = 'min';
        dMode = 'max';
        compMode = 'minVwithL';
        data0 = targets;
        HJIextraArgs.obstacleFunction = obstacles;
        HJIextraArgs.targetFunction = targets;
    elseif strcmp(v1_mode, 'bras')
        uMode = 'min';
        dMode = 'max';
        compMode = 'none';
        data0 = targets;
        HJIextraArgs.obstacleFunction = obstacles;
    else 
        error('Unknown v1_mode');
    end

    schemeData.grid = grid;
    schemeData.dynSys = dyn_sys;
    schemeData.accuracy = 'veryHigh'; 
    schemeData.uMode = uMode;
    schemeData.dMode = dMode;

    HJIextraArgs.visualize.valueSet = 1;
    HJIextraArgs.visualize.initialValueSet = 1;
    HJIextraArgs.visualize.figNum = 1;
    HJIextraArgs.visualize.deleteLastPlot = true;
    
    HJIextraArgs.visualize.plotData.plotDims = [1 1];
    HJIextraArgs.visualize.plotData.projpt = [];
    HJIextraArgs.visualize.viewAngle = [0,90];
    % data0
    [data, ~, ~] = ...
        HJIPDE_solve(data0, tau, schemeData, compMode, HJIextraArgs);
    g_with_time = createGrid([grid.min; tau(1)], [grid.max; tau(end)], [grid.shape, length(tau)]);
    deriv_v1 = computeGradients(g_with_time, data);
    opt_v1_ctrl = dyn_sys.optCtrl([], [], computeGradients(grid, data), schemeData); 
end
