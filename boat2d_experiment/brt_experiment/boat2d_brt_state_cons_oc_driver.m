% This driver script defines the grid, dynSys, time horizon, obstacle
% READ BEFORE PROCEEDING: need to define optCtrl for the dynSys, and the
% custom ham func. Depending on the running cost structure, two functions
% might need to be modified. 

%%
clear; close all; clc;

%% define grid and time horizon
grid_min = [-3., -2.25, -1];
grid_max = [2.25, 2.25, 20];
N = [70, 70, 210];
grid = createGrid(grid_min, grid_max, N);

t0 = 0;
tMax = 2;
dt = 0.05;
tau = t0:dt:tMax;

%% define running cost and final cost 
goal_x = 1.5; 
goal_y = 0; 

% final_cost_func_handle = @(state) ((state{1} - goal_x).^2 + (state{2} - goal_y).^2).^0.5;
final_cost_func_handle = @(state) 0.; 
running_cost_func_handle_cell = @(state,ctrl) ((state{1} - goal_x).^2 + (state{2} - goal_y).^2).^0.5; 
running_cost_func_handle_arr = @(state,ctrl) ((state(1) - goal_x).^2 + (state(2) - goal_y).^2).^0.5; 


%% define dynSys
a = 0.5;
c = 2;
vMax = 1.; 
dyn_sys = Boat2DAugBRT([0;0], vMax, c, a, running_cost_func_handle_cell,running_cost_func_handle_arr);

%% define obstacles and target
obs1 = shapeRectangleByCenter(grid, [-0.5, 0.5, 0], [0.8, 0.8, inf]);
obs2 = shapeRectangleByCenter(grid, [-1, -1.25, 0], [0.4, 1.5, inf]);
obs3 = shapeHyperplane(grid, [-1;0;0], [2;0;0]);
obs4 = shapeHyperplane(grid, [0;-1;0], [0;2;0]);
obs5 = shapeHyperplane(grid, [0;1;0], [0;-2;0]);
obs6 = shapeHyperplane(grid, [1;0;0], [-3;0;0]);
obs = shapeUnion(obs1, obs2);
obs = shapeUnion(obs, obs3);
obs = shapeUnion(obs, obs4);
obs = shapeUnion(obs, obs5);

%% call template function
v1_mode = 'brt_avoid'; 
extra_args.use_analytical_opt_ctrl = true;
% extra_args.custom_ham = @boat2DStateConstrainedHamBRT;
[data_vfunc_aux] = computeStateConsValueFunc(grid, dyn_sys, obs, [], final_cost_func_handle, tau, v1_mode, extra_args);
deriv_vfunc_aux = computeGradients(grid, data_vfunc_aux);
[data_vfunc, min_z_index_arr] = AuxVFunc2VFunc(grid, data_vfunc_aux, tau);
close all; figure; axis equal; 
surf(grid.xs{1}(:,:,1), grid.xs{2}(:,:,1), data_vfunc(:,:,end)); 
title('State Constrained Formulation Value Function'); 
save('experiments_data/boat2d_brt_state_cons_data_210.mat', 'data_vfunc_aux',...
    'data_vfunc', 'min_z_index_arr', 'deriv_vfunc_aux', 'grid', 'tau', 'dyn_sys', '-v7.3');

%% custom ham func for running cost = l2 dist to goal
function hamValue = boat2DStateConstrainedHamBRT(t, data, deriv, schemeData)
    deriv_x_y_l2_norm = (deriv{1} .^ 2 + deriv{2} .^ 2) .^ 0.5;
    hamCtrl = (schemeData.dynSys.c - schemeData.dynSys.a .* schemeData.grid.xs{2} .^ 2) .* deriv{1} ... 
        - deriv_x_y_l2_norm .* schemeData.dynSys.vMax;
    hamRunning = - deriv{3} .* schemeData.dynSys.running_cost_func_handle_cell(schemeData.grid.xs, 0);
    hamValue = hamCtrl + hamRunning;

    if strcmp(schemeData.tMode, 'backward')
      hamValue = -hamValue;
    else
      error('tMode must be ''backward''!')
    end

end