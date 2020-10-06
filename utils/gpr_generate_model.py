"""
Copyright 2020, Zi Xian Leong (Author). All rights reserved.

Email: zxleong@psu.edu
"""

#%%

'''Generates "wavy" layers of subsurface models '''

'''The parameters available:

    num_vel = 500; \\ this is the number of models you want to generate
    air = 10; \\ first 10 rows are air layer, i.e. velocity of air = 299792500 m/s
    num_layers = (4,6); \\ random number of layers from 4 to 5. Is 4 - 5 because of python norm
    num_bricks_each_layer = (4,9); \\ for each layer, random number of bricks
    layer_grid_fluc = 3; \\ controls how much each spline points move vertically
    height = 128; \\ height of model
    width = 128; \\ width of model
    x_var = (0,128); \\ controls the x coordinates of spline points
    order = (2,5); \\ controls the polynomial order fit between the spline points
    vel_min = 48; \\ minimum velocity is 48e6
    vel_max = 175; \\ maximum velocity is 175e6
    
    
'''

#%%
import numpy as np

# %%
def generate_interface_gpr(num_vel=500, air=10, num_layers=(4,6), num_bricks_each_layer=(4,9),
                  layer_grid_fluc = 3, height = 128, width=128,
                   x_var=(0,128), order=(2,5),  vel_min=48, vel_max=175):

    # Layers start from specified depth (usually following water/air depth)
    start_from = 0

    #Specify canvas size
    vertical_dim = height
    horizontal_dim = width
    
    # Empty stack to store velocity values
    vel_stack = []
    for each_vel in (range(num_vel)):

        #random number of layers
        num_layers_ = np.random.randint(num_layers[0],num_layers[1])
        
        # Thickness of each layer
        bricks_height = int(np.ceil((vertical_dim-start_from)/num_layers_))
        
        # Compute vertical axis of indices of layers
        ls_layers_ind = np.arange(start_from+bricks_height, vertical_dim, bricks_height) #y index values of velocities
        
        ls_layers_ind = np.append(ls_layers_ind, vertical_dim+5) #add final index
    
        #Create spline mask that returns dict of indices of spline interfaces
        xy_spline_ind = dict()
        for row, interface_i in zip(ls_layers_ind, range(len(ls_layers_ind))):
            row = int(row)
                
            th = layer_grid_fluc
            # compute random x and y coordinates for spline  
            ydots = np.random.randint(row-th, row+th, size=20)
            xdots = np.random.randint(x_var[0],x_var[1], size = 20)
            
            #Order of spline fitting
            deg = np.random.randint(order[0],order[1])
            coeff = np.polyfit(xdots,ydots,deg)
            
            #Compute y axis indices of spline curve
            y_spline_ind=[]
            for j in range(0, horizontal_dim):
                val = int(round(np.polyval(coeff, j)))
        
                #force limit y index's maximum value to be 127
                if val > vertical_dim-1:
                    y_spline_ind.append(vertical_dim-1)
                else:
                    y_spline_ind.append(val)
                    
                
            #Store x & y coordinates of wavy interfaces
            y_spline_ind = np.array(y_spline_ind)
            x_spline_ind = np.arange(0, horizontal_dim)
            xy_spline_ind['interface{}'.format(interface_i)] = x_spline_ind, y_spline_ind
                        
    
        # Function to find y coordinates between two y points
        def get_y_coordinates( y_coord_top, y_coord_bot):
            y_arr = []
            for i,j in zip(y_coord_top, y_coord_bot):
                y_arr.append(np.arange(i, j+1))
            return y_arr
        
        #Create an array
        tile = np.ones((vertical_dim,horizontal_dim))
    
        #After creating mask of indices, we need to fill the shape with velocity values
        start_interf = 0
        for each_interf in range(1,len(ls_layers_ind)):
            
            #Specify top interface
            x_arr_top = xy_spline_ind['interface{}'.format(start_interf)][0]
            y_arr_top = xy_spline_ind['interface{}'.format(start_interf)][1]
            
            #Specify immediate bottom interface
#            x_arr_bot = xy_spline_ind['interface{}'.format(each_interf)][0]
            y_arr_bot = xy_spline_ind['interface{}'.format(each_interf)][1]
            
            # Number of bricks in each layer
            num_bricks = np.random.randint(num_bricks_each_layer[0],num_bricks_each_layer[1]) 
            
            
            brick_interval = int(horizontal_dim/num_bricks)
            ls_horizontal_ind = np.ceil(np.linspace(brick_interval,horizontal_dim,num_bricks))
            
            #Filling in the bricks
            col_start = 0
            for col_edges in ls_horizontal_ind:
                col_edges = int(col_edges)
                x_arr_top_cut = x_arr_top[col_start:col_edges]
                y_arr_top_cut = y_arr_top[col_start:col_edges]
                
                 #'''same as x_arr_top_cut because it has all x values'''
#                x_arr_bot_cut = x_arr_bot[col_start:col_edges] 
                y_arr_bot_cut = y_arr_bot[col_start:col_edges]
                
                #Finds ALL the coordinates that are in each brick
                x_arr = x_arr_top_cut
                y_arr = get_y_coordinates(y_arr_top_cut, y_arr_bot_cut)
                
                all_x_coordinates = []
                all_y_coordinates = []
                for i in range(len(y_arr)):
                    
                    x_val = x_arr[i]
                
                    if x_val <0:
                        x_val = 0
                    
                    for j in range(len(y_arr[i])):
                        y_val = y_arr[i][j]
                        
                        if y_val <0:
                            y_val=0
                        
                        all_x_coordinates.append(x_val)
                        all_y_coordinates.append(y_val)
            
                #Compute the velocity value from the initial model and input in to new array
                # new_vel = vel_condition(np.mean(init_model[all_y_coordinates, all_x_coordinates]))
                vel_rand = np.random.randint(vel_min,vel_max)
                tile[all_y_coordinates, all_x_coordinates] = vel_rand * 1e6
            
             
                col_start += brick_interval                
            start_interf += 1
            tile[0:air,:] = 299792500
    
        vel_stack.append(tile)
            
    vel_stack = np.array(vel_stack)
    return vel_stack
