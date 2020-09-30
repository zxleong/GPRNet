"""
Copyright 2020, Zi Xian Leong (Author). All rights reserved.

Email: zxleong@psu.edu
"""



import numpy as np
import random
from numpy.matlib import repmat

def thickness_block(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    ''' Source: https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value '''

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

  
def vel_generator(gamble=0,velmin=48,velmax=200, depth=105, thickmin=4, thickmax=18):
    """Return a 1D list of velocity values with random number of layers, and the values are in descending order
        gamble = 0; \\ randomize it 0 or 1
        velmin = 48; \\ minimum velocity from randomization
        velmax= 200; \\ maximum velocity from randomization
        depth = 105; \\ number of grid cell, i.e. 105 grid cells
        thickmin = 4; \\ minimum number of layers
        thickmax = 18; \\ maxmimum number of layers
    
    """
    
    thick_rand = np.random.randint(thickmin,thickmax) # layers
    # total_depth = np.random.randint(int(85/diffdepth),int(98/diffdepth))  #85 - 98
    total_depth = depth
    thicc_layers = thickness_block(thick_rand, total_depth)
    vel_profile = np.array([])
    vel_rand = np.random.randint(velmin,velmax, size=thick_rand ) #this is in 1e6
    
    if gamble == 0:
        vel_rand = sorted(vel_rand, reverse=True)
    elif gamble ==1:
        pass    
    
    
    rnum1 = np.random.randint(1, len(vel_rand)-1)
    rnum2 = []
    while len(rnum2) < 1:
        num = np.random.randint(1, len(vel_rand)-1)
        if num not in [rnum1]:
            rnum2.append(num)
    rnum3 = []
    while len(rnum3) < 1:
        num = np.random.randint(1, len(vel_rand)-1)
        if num not in [rnum1] and num not in [rnum2]:
            rnum3.append(num)
    
    vel_rand[rnum1], vel_rand[rnum2[0]] = vel_rand[rnum2[0]], vel_rand[rnum1]
    vel_rand[rnum1], vel_rand[rnum3[0]] = vel_rand[rnum3[0]], vel_rand[rnum1]
    
    num_jumps = np.random.randint(1,15)
    rnum = np.random.randint(1, len(vel_rand), size=num_jumps)
    for i in rnum:
        random_value = np.random.randint(20,50) #random velocity value to be added
        vel_rand[i] += random_value
    
    
    for j,k in zip(thicc_layers, vel_rand):
    
        each_layer = repmat(k, 1, j)
        each_layer = np.ndarray.flatten(each_layer)
        vel_profile = np.concatenate([vel_profile, each_layer])
    
    return vel_profile
