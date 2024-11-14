function [vfunc, min_z_index_arr] = AuxVFunc2VFunc(grid, data, tau)
% Inputs
%   grid: grid for the auxiliary value function
%   data: the auxiliary value function, dimension (state dim + 1, time dim)
%   tau: time steps vector of size (1, num time steps)
% Outputs
%   vfunc: actual value function, dimemsion (state dim, time dim)
%   min_z_index_arr: the index of z^*, dimension same as vfunc
%   Important Note: typically vfunc is parameterized by (t,x), but here we 
%   do (x,t) following helperOC's convention

    g = grid;   
    state_dim = grid.dim - 1;
    t_dim = size(tau, 2);
    if state_dim == 2
        vfunc = zeros(g.N(1), g.N(2), t_dim);
        min_z_index_arr = zeros(g.N(1), g.N(2), t_dim); 
        tic
        for t_index = 1:1:t_dim
            % current_time = tau(t_index);
            for i=1:1:g.N(1)
                for j=1:1:g.N(2)
                    min_z = g.max(3);
                    min_z_index = g.N(3); 
                    for k = 1:1:g.N(3)
                        current_z = g.vs{3}(k);
                        if data(i,j,k, t_index) <= 0
                            min_z = current_z; 
                            min_z_index = k;
                            break;
                        end
                    end
                vfunc(i, j, t_index) = min_z;
                min_z_index_arr(i, j, t_index) = min_z_index; 
                end
            end
        end
        disp('Compute Value Function from Aux Value Function Time Taken');
        toc
    elseif state_dim == 3
        vfunc = zeros(g.N(1), g.N(2), g.N(3));
        min_z_index_arr = zeros(g.N(1), g.N(2), g.N(3)); 
        error('Not Implemented');
    elseif state_dim == 4
        vfunc = zeros(g.N(1), g.N(2), g.N(3), g.N(4)); 
        min_z_index_arr = zeros(g.N(1), g.N(2), g.N(3), g.N(4)); 
        error('Not Implemented');
    else
        error('Undefined state dimension'); 
    end


end