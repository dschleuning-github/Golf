#TODO...........First image
    #...y_grid,x_grid  #TODO only on first image?
    #...find the ball, clean up extra LEDS.
#TODO...........find the club head
#TODO...........calculate spin!!!
#####################################################

from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from image_analysis_v2_0 import remove_background, raster_ball, hill_climb_master, spin_find
import GolfFlight_v2_0

##################################################
########### Set the Path ########
#path_dir='20220402_Blue'
#foreground_LED=False

#path_dir='20220420_Droom1'
#ball_radius = 25   #25
#foreground_LED=True
#x_LED_o = 95
#y_LED_o = 135
#delta_LED=30

path_dir='20220422_Golfbay3'
ball_radius = 25   #25
foreground_LED=True
x_LED_o = 10
y_LED_o = 160
delta_LED=30

############ Some Global constants ##########
Ball_diameter_mm = 42.0
Ball_diameter_pix = 60
mm_per_pix = Ball_diameter_mm/Ball_diameter_pix

global y_temp,x_temp   #TODO...one time?

############### START ###########
file_list = os.listdir(path_dir)
file_list.sort()
file_list_len = len(file_list)
print(file_list)  #['out.000108.raw.tiff', out.000109.raw.tiff'....]

fig,ax = plt.subplots(6,2, figsize=(9,8))
image_data_list=[]
f_index=0
f_index_list=[]
for f in np.arange(len(file_list)):
    t0 = time.time()
    file_name = file_list[f]    
    start_img = Image.open(os.path.join(path_dir, file_name))
    start_np = np.array(start_img)   #print(start_np.shape)   #(170, 400, 3)
    blue_np = start_np[:,:,2]       #blue_img = Image.fromarray(blue_np)

#    blue_np, back_stats = remove_background(blue_np)
#    blue_np, back_stats = image_analysis.remove_background(blue_np,
    blue_np, back_stats = remove_background(blue_np,                                                           
            foreground_LED, x_LED_o, y_LED_o, delta_LED)

######### Raster using radius mask ########
    y_grid,x_grid = np.ogrid[0:blue_np.shape[0], 0: blue_np.shape[1]]  #TODO only on first image?
    Xball, Yball = raster_ball(blue_np, y_grid,x_grid, ball_radius)    
#    print(f, 'raster', Xball, Yball)

##########  START Hill Climb  ###############
    Xball, Yball, temp_sum = hill_climb_master(blue_np,
                Xball, Yball, ball_radius,y_grid,x_grid )
#    print('....hill climb', Xball, Yball)
#    temp_sum =1   #TODO...need to fix the V,A Calc
    if f==0:
        Xball_o = Xball
        Yball_o = Yball
    elif (f_index<3):
        if (math.sqrt( (Xball-Xball_o)**2 + (Xball-Xball_o)**2) > ball_radius):
            f_index = f_index + 1
            theta_back, theta_yaw = spin_find(blue_np,
                Xball, Yball, ball_radius,y_grid,x_grid)
            print(f, f_index, theta_back, theta_yaw)
            f_index_list.append([f_index, f])
            Xwidth=Ywidth=ball_radius
            final_img = Image.fromarray(blue_np)
            bbox =  (Xball - Xwidth, Yball - Ywidth, Xball + Xwidth, Yball + Ywidth)
            draw = ImageDraw.Draw(final_img)
            xxx= draw.ellipse(bbox, outline=255)
            ax[f_index,0].imshow(start_img)
            ax[f_index,1].imshow(final_img, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'
            ax[f_index,1].text(20, 20, 'frame = {}'.format(f))
            ax[f_index,1].text(20, 40, 'x, y = {}, {}'.format(Xball,Yball), fontsize=8)
            ax[f_index,1].text(20, 60, 'median={:.0f}, std={:.1f}'.format(back_stats[0],back_stats[1]),fontsize=8)
        else:
            pass
    else:
        break
    print(f, f_index, Xball, Yball, time.time()-t0)
    image_data_list.append([f, Xball, Yball, temp_sum, 0,0,0])
print(f_index_list)   
print('image_data =', image_data_list)

########## Plot the pre-hit frame #########
f_number = f_index_list[0][1] -1
file_name = file_list[f_number]    
start_img = Image.open(os.path.join(path_dir, file_name))
start_np = np.array(start_img)
blue_np = start_np[:,:,2]         #blue_img = Image.fromarray(blue_np)
blue_np, back_stats = remove_background(blue_np,  #)
            foreground_LED, x_LED_o, y_LED_o, delta_LED)

img_blue = Image.fromarray(blue_np)
final_img = Image.fromarray(blue_np)
if (image_data_list[f_number][0] != f_number):
    print('whoops....which file am I on?')
Xball = image_data_list[f_number][1]
Yball = image_data_list[f_number][2]
bbox =  (Xball - Xwidth, Yball - Ywidth, Xball + Xwidth, Yball + Ywidth)
draw = ImageDraw.Draw(final_img)
xxx= draw.ellipse(bbox, outline=255)
ax[0,0].imshow(start_img)
ax[0,1].imshow(final_img, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'
ax[0,1].text(20, 20, 'frame = {}'.format(f_number))
ax[0,1].text(20, 40, 'x, y = {}, {}'.format(Xball,Yball), fontsize=8)
ax[0,1].text(20, 60, 'median={:.0f}, std={:.1f}'.format(back_stats[0],back_stats[1]),fontsize=8)

#########################################
#......TODO....plot line, use multiple ball measurement to "fit" ball v, angle; 
time_list=[]
x0 = image_data_list[0][1]
y0 = image_data_list[0][2]
x_list = []
y_list = []
T_pulse = 2 #mS
for f in np.arange(f_index_list[0][1] -1,f_index_list[2][1]+1 ):
    if(image_data_list[f][3] != 0):
        time_list.append(image_data_list[f][0]*T_pulse)
        x_list.append(image_data_list[f][1] - x0)
        y_list.append(-(image_data_list[f][2] - y0))
print('x_list = ',x_list)
if (len(x_list) > 2):
    launch_angle_deg =  180/math.pi*math.atan2((y_list[2] - y_list[1]),(x_list[2] - x_list[1]) )
    Distance_pix = math.sqrt( (y_list[2] - y_list[1])**2 + (x_list[2] - x_list[1])**2)
    launch_velocity_mps = mm_per_pix*Distance_pix/2  #2mS flash period
    launch_velocity_MPH = launch_velocity_mps*3600/1609.34
else:
    launch_angle_deg = 0
    launch_velocity_mps =0

#print('velocity ={:.2f}m/s ({}MPH'.format(launch_velocity_mps, launch_velocity_MPH))
#print('angle = {:.1f} deg)'.format(launch_angle_deg) ) 
    
ax[4,0].scatter(x_list, y_list)
ax[4,0].set_xlim(0,150)
ax[4,0].set_ylim(-10,65)
ax[4,0].set_xlabel('x pixels')
ax[4,0].set_ylabel('y pixels')
ax[4,0].text(10,50, 'V={:.1f}m/s ({:.1f}MPH)'.format(launch_velocity_mps,
                            launch_velocity_MPH) )
ax[4,0].text(10,30, 'A={:.1f} deg'.format(launch_angle_deg) )

###############################################################
##   Start Max's Section  ###############################
#############################################################

xxx, yyy, carry, height = GolfFlight_v2_0.calculateDistance(launch_velocity_mps,
                    launch_angle_deg*math.pi/180, 7000)
ax[5,0].plot(xxx,yyy)
ax[5,0].text(50,10, 'carry={:.1f} yards'.format(carry) )
ax[5,0].text(50,5, 'height={:.1f} yards '.format(height) )

plt.show()

#######################################################
######  BACKUP      ##################################
#######################################################

