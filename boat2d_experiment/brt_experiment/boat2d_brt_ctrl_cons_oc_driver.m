clear; close all; clc;

warning on all
warning on backtrace
warning on verbose

t0 = 0;
tMax = 2;
dt = 0.05;
tau = t0:dt:tMax;

vMax = 1;
dMax = 0;
goalX = 1.5;
goalY = 0;
goalR = 0.25;
a = 0.5;
c = 2;

% define grid
grid_min = [-3., -2.25];
grid_max = [2.25, 2.25];
N = [70, 70];
grid = createGrid(grid_min, grid_max, N);

obs1 = shapeRectangleByCenter(grid, [-0.5, 0.5], [0.8, 0.8]);
obs2 = shapeRectangleByCenter(grid, [-1, -1.25], [0.4, 1.5]);
obs3 = shapeHyperplane(grid, [-1;0], [2;0]);
obs4 = shapeHyperplane(grid, [0;-1], [0;2]);
obs5 = shapeHyperplane(grid, [0;1], [0;-2]);
obs = shapeUnion(obs1, obs2);
obs = shapeUnion(obs, obs3);
obs = shapeUnion(obs, obs4);
obs = shapeUnion(obs, obs5);

v1_mode = 'brt_avoid';
boat1 = Boat2D([0;0], vMax, c, a);
[data_v1, deriv_v1, opt_ctrl_v1] = computeV1(grid, boat1, obs, [], tau, v1_mode);

terminal_cost = @(x) 0. .* x{1};
running_cost = @(x,u) L(x,u);
accuracy = 'veryHigh'; 
compute_best_constraints = true;

boat2 = Boat2DCtrlConsBRT([0;0], vMax, c, a, running_cost);
use_analytical_optimal_control = true; 
v1_threshold = 0.03; 
[data_v2] = computeCtrlConsValueFunc(grid, boat2, tau, v1_mode,...
             data_v1, deriv_v1, opt_ctrl_v1, running_cost, terminal_cost, ...
             compute_best_constraints, use_analytical_optimal_control, accuracy, v1_threshold);

deriv_v2 = computeGradients(grid, data_v2);

save('experiments_data/boat2d_brt_ctrl_cons_data_03', 'data_v2', 'deriv_v2', 'grid', 'tau', 'boat1', 'boat2',...
    'data_v1', 'deriv_v1', 'opt_ctrl_v1', 'goalX', 'goalY', 'goalR', 'running_cost', 'v1_threshold');
        
%%
function cost = L(x, u)
    if ~iscell(x)
        x = num2cell(x);
    end
    if ~iscell(u)
        u = num2cell(u);
    end
    cost = ((x{1} - 1.5).^2 + x{2}.^2).^0.5;
end

