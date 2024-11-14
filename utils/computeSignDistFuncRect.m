function sign_dist = computeSignDistFuncRect(points, rect)
    center_x = rect(1);
    center_y = rect(2);
    width = rect(3);
    height = rect(4);

    left = center_x - width / 2; 
    right = center_x + width / 2; 
    top = center_y + height / 2; 
    bottom = center_y - height / 2; 
    
    px = points(1,:);
    py = points(2,:);

    dx = max(max(left - px, px - right), zeros(1, size(points, 2)));
    dy = max(max(bottom - py, py - top), zeros(1, size(points, 2))); 

    outside_dist = (dx.^2 + dy.^2).^0.5;
    inside_dist_x = min(px - left, right - px);
    inside_dist_y = min(py - bottom, top - py);
    inside_dist = min(inside_dist_x, inside_dist_y);

    inside_mask = (left <= px) & (px <= right) & (bottom <= py) & (py <= top);
    outside_mask = ~ inside_mask; 

    sign_dist = -inside_dist .* inside_mask + outside_dist .* outside_mask;


end