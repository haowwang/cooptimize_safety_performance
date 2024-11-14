function sign_dist = computeSignDistBoatEnv(points)
    right_plane_normal = [-1;0]; 
    right_plane_anchor = [2;0]; 
    top_plane_normal = [0;-1];
    top_plane_anchor = [0;2]; 
    bottom_plane_normal = [0;1]; 
    bottom_plane_anchor = [0;-2];

    sign_dist_right_plane = computeSignDistFuncHP(right_plane_normal, ...
        right_plane_anchor, points);
    sign_dist_top_plane = computeSignDistFuncHP(top_plane_normal, ...
        top_plane_anchor, points);
    sign_dist_bottom_plane = computeSignDistFuncHP(bottom_plane_normal, ...
        bottom_plane_anchor, points);
    
    rect_1 = [-0.5, 0.5, 0.8, 0.8]; % center_x, center_y, full width, full height
    rect_2 = [-1, -1.25, 0.4, 1.5]; 

    sign_dist_rect_1 = computeSignDistFuncRect(points, rect_1); 
    sign_dist_rect_2 = computeSignDistFuncRect(points, rect_2);

    sign_dist = min(sign_dist_rect_1, sign_dist_rect_2);
    sign_dist = min(sign_dist, sign_dist_right_plane); 
    sign_dist = min(sign_dist, sign_dist_top_plane); 
    sign_dist = min(sign_dist, sign_dist_bottom_plane); 

    if size(sign_dist, 1) > 1 
        error('Incorrect dimension in signed distance computation');
    end

end