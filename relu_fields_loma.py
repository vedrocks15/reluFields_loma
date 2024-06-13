# Helper function to compute the output of ReLU-Fields....
def reluField2d(feature_grid :  In[Array[Array[float]]], 
                pos_var : In[Array[float]],
                grid_size : In[Array[float]]) -> float:
    
    
    # Function that takes as input the learnt feature grid and output the 
    # bi-linearly interpolated feature value at the position location.
    # Arguments :
    #   feature_grid : 2D array of floats containing the feature values.
    #   pos_var   : location at which we want the feature value (tuple of float values) (y,x)
    #   grid_size : size of the feature grid (h,w)
    # Returns :
    #   Bilinealy interpolated feature value....
    
    # 1. defining floor function
    pos_var_g_x : int = pos_var[1]
    pos_var_g_y : int = pos_var[0]

    # 2. extracting the fractional part 
    pos_var_frac_x : float = pos_var[1] - pos_var_g_x
    pos_var_frac_y : float = pos_var[0] - pos_var_g_y
    
    # 3. offset locations for bilinear interpolation
    y0 : int = pos_var_g_y + 0
    x0 : int = pos_var_g_x + 0
    y1 : int = pos_var_g_y + 1
    x1 : int = pos_var_g_x + 1
    
    # 4. grid size checks
    if grid_size[0] <= y1:
        y1 = grid_size[0] - 1
    if grid_size[1] <= x1:
        x1 = grid_size[1] - 1    
    
    # 5. extracting the 4 co-ordinates on 2d grid for Bi-Linear Interpolation
    feat_00 : float = feature_grid[y0][x0]
    feat_01 : float = feature_grid[y0][x1]
    feat_10 : float = feature_grid[y1][x0]
    feat_11 : float = feature_grid[y1][x1]

    # 6. Bi-linear interpolation....
    # 6.1) along x - axis....
    temp_row_0 : float = feat_01*pos_var_frac_x + (1-pos_var_frac_x)*feat_00
    temp_row_1 : float = feat_11*pos_var_frac_x + (1-pos_var_frac_x)*feat_10

    # 6.2) along y - axis.....
    interploated_feature : float = temp_row_1*pos_var_frac_y + (1-pos_var_frac_y)*temp_row_0

    # 7. Apply relu function.....
    if interploated_feature < 0:
        return 0.0
    else:
        # maximum clipping....
        if interploated_feature >= 255.0:
            return 1.0
        else:
            # Normalizing values.....
            return interploated_feature/255.0

# Computing the reverse diff graph for the function....
d_rField_diff = rev_diff(reluField2d)

