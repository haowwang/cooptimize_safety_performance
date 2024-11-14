function sign_dist = computeSignDistFuncHP(normal, anchor, points)
% This function computes the sign distance function to a hyperplane
% Inputs: 
%   normal: the normal of the hyperplane (column vector)
%   anchor: a point the hyperplane passes through (column vector)
%   points: points that we want to compute the sign dist for (horizontally
%   stacked column vectors)

    if ndims(normal) > 2 | ndims(anchor) > 2 | ndims(points) > 2
        error('Some inputs are not in required column vector form');
    end

    if size(normal,2) > size(normal,1)
        normal = normal';
    end
    
    if size(anchor,2) > size(anchor,1)
        anchor = anchor';
    end
    
    
    if ~ (length(normal) == length(anchor))
        error('Normal and anchor of hyperplane have different dimensions');
    end

    if ~ (length(normal) == length(points(:,1)))
        error('Normal of hyperplane and points have different dimensions'); 
    end

    normal = repmat(normal, [1, size(points, 2)]);
    anchor = repmat(anchor, [1, size(points, 2)]); 

    sign_dist = dot(normal, points - anchor, 1);

end