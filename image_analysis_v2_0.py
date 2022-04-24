import numpy as np

def remove_background(xxx_np,foreground, x_o, y_o, delta):
    ###...Remove the foreground LEDs & Background
    if foreground:
        x_LED = [x_o, x_o+30, x_o+65, x_o+105,x_o+125] #[160, 195, 240, 290, 325]
        y_LED = [y_o,y_o,y_o-5,y_o-10, y_o-10] #[130, 130, 125, 120, 120]
        for i in np.arange(len(x_LED)):
            xxx_np[y_LED[i]:y_LED[i]+delta, x_LED[i]:x_LED[i]+delta]=0 #.....Mask
    else:
        pass

    ###...Remove the "background"""
    blue_mean = xxx_np.mean()
    blue_median = np.median(xxx_np)
    blue_std = xxx_np.std()
    xxx_np[xxx_np < (blue_median + 1*blue_std)]=0      #...............................Mask
    background_stats =[blue_median, blue_std]
    return xxx_np,background_stats

def raster_ball(blue_np, y_temp, x_temp, ball_r):
    ball_dict={}
    for x in np.arange(0, blue_np.shape[1],ball_r):  #10
        for y in np.arange(0,blue_np.shape[0],ball_r):
            temp_np = blue_np.copy()
            mask = (x-x_temp)**2+(y-y_temp)**2 <= ball_r**2
            temp_sum = temp_np[mask].sum()
            ball_dict[x,y] = temp_sum
    max_val = list(ball_dict.values())
    max_ke = list(ball_dict.keys())
    Xball, Yball = max_ke[max_val.index(max(max_val))]
    return Xball, Yball

def spin_find(data_np, Xball, Yball, ball_r, y_temp,x_temp):
    spin_dict={}
    for a in np.arange(0, 360, 20):
        temp_np = data_np.copy()
        mask = ((Xball-x_temp)**2+(Yball-y_temp)**2 <= ball_r**2) | (a==40)
        temp_sum = temp_np[mask].sum()
        spin_dict[a] = temp_sum
    max_val = list(spin_dict.values())
    max_ke = list(spin_dict.keys())
    theta_back = max_ke[max_val.index(max(max_val))]        

#    theta_back = 0
    theta_yaw=0
    return theta_back, theta_yaw


def hill_climb(dx, dy, data_np, Xball, Yball, ball_r, y_temp,x_temp):
        #TODO could y_temp, x_temp be some type of global?
    hill_dict={}
    hill_dict[Xball,Yball]= 1
    climb_status = True
    while climb_status:
        x = Xball + dx
        y = Yball + dy
        temp_np = data_np.copy()
        mask = (x-x_temp)**2+(y-y_temp)**2 <= ball_r**2
        temp_sum = temp_np[mask].sum()
        hill_dict[x,y] = temp_sum
        max_val = list(hill_dict.values())
        max_ke = list(hill_dict.keys())
        if  temp_sum > max(max_val[:-1]):
            Xball = x
            Yball = y
        else:
            climb_status=False   
    return Xball, Yball, temp_sum
   
def hill_climb_master(data_np, Xball, Yball, ball_r, y_temp,x_temp):
    Xball, Yball, temp_sum = hill_climb(1,0, data_np, Xball, Yball, ball_r, y_temp,x_temp)
    Xball, Yball, temp_sum = hill_climb(0,1, data_np, Xball, Yball, ball_r, y_temp,x_temp)
    Xball, Yball, temp_sum = hill_climb(-1,0, data_np, Xball, Yball, ball_r, y_temp,x_temp)
    Xball, Yball, temp_sum = hill_climb(0,-1, data_np, Xball, Yball, ball_r, y_temp,x_temp)
    Xball, Yball, temp_sum = hill_climb(1,0, data_np, Xball, Yball, ball_r, y_temp,x_temp)
    Xball, Yball, temp_sum = hill_climb(0,1, data_np, Xball, Yball, ball_r, y_temp,x_temp)
    return Xball, Yball, temp_sum




######### BACKUP ################

#import matplotlib.pyplot as plt
#from scipy.signal import find_peaks, peak_widths
#import math
#from PIL import Image, ImageDraw
#import os
