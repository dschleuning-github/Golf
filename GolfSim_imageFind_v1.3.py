from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import math
#from math import sqrt
#import pandas as pd

##################################################
########### Set the Path ########
#path_dir='20220402_Blue'
#foreground_LED=False

#path_dir='20220403_Golfbay1'
#foreground_LED=True
#start_frame =0
#end_frame = 4         
#ball_radius = 29

path_dir='20220416_Droom5'
foreground_LED=True
ball_radius = 20   #25
###################################################


############ Some Global constants ##########
Ball_diameter_mm = 42.0
Ball_diameter_pix = 60
mm_per_pix = Ball_diameter_mm/Ball_diameter_pix

global y_temp,x_temp

#####################################
##########  Define Functions ########
#####################################

###...Remove the foreground LEDs & Background
def remove_background(xxx_np):
    ###...Remove the "LEDs"
    x_LED_o = 95
    y_LED_o = 135
    delta_LED=30
    if foreground_LED:
        x_LED = [x_LED_o, x_LED_o+30, x_LED_o+65, x_LED_o+105,x_LED_o+125] #[160, 195, 240, 290, 325]
        y_LED = [y_LED_o,y_LED_o,y_LED_o-5,y_LED_o-10, y_LED_o-10] #[130, 130, 125, 120, 120]
        for i in np.arange(len(x_LED)):
            xxx_np[y_LED[i]:y_LED[i]+delta_LED, x_LED[i]:x_LED[i]+delta_LED]=0 #.....Mask
    else:
        pass
    ###...Remove the "background"""
    blue_mean = xxx_np.mean()
    blue_median = np.median(xxx_np)
    blue_std = xxx_np.std()
    xxx_np[xxx_np < (blue_median + 1*blue_std)]=0      #...............................Mask
    background_stats =[blue_median, blue_std]
    return xxx_np,background_stats

def raster_ball(blue_np):


    ball_dict={}
    for x in np.arange(0, blue_np.shape[1],10):
        for y in np.arange(0,blue_np.shape[0],10):
            temp_np = blue_np.copy()
            mask = (x-x_temp)**2+(y-y_temp)**2 <= ball_radius**2
            temp_sum = temp_np[mask].sum()
            ball_dict[x,y] = temp_sum
    max_val = list(ball_dict.values())
    max_ke = list(ball_dict.keys())
    Xball, Yball = max_ke[max_val.index(max(max_val))]
#    print('raster =', Xball, Yball, ball_radius)
    return Xball, Yball

def hill_climb(dx, dy):
    climb_status = True
    global Xball, Yball, temp_sum
    while climb_status:
        x = Xball + dx
        y = Yball + dy
        temp_np = blue_np.copy()
#    y_temp,x_temp = np.ogrid[0:len(Ysum), 0: len(Xsum)]
        mask = (x-x_temp)**2+(y-y_temp)**2 <= ball_radius**2
        temp_sum = temp_np[mask].sum()
        hill_dict[x,y] = temp_sum
        max_val = list(hill_dict.values())
        max_ke = list(hill_dict.keys())
        if  temp_sum > max(max_val[:-1]):
            Xball = x
            Yball = y
        else:
            climb_status=False


file_list = os.listdir(path_dir)
file_list.sort()
file_list_len = len(file_list)
print(file_list)  #['out.000108.raw.tiff', out.000109.raw.tiff'....]

fig,ax = plt.subplots(5,2, figsize=(9,8))
image_data_list=[]
f_index=0
#f_index_flag = False
f_index_list=[]
for f in np.arange(len(file_list)):
    file_name = file_list[f]    
    start_img = Image.open(os.path.join(path_dir, file_name))
    start_np = np.array(start_img)   #print(start_np.shape)   #(170, 400, 3)
    blue_np = start_np[:,:,2]       #blue_img = Image.fromarray(blue_np)

    blue_np, back_stats = remove_background(blue_np)

######### Raster using radius mask ########
    y_temp,x_temp = np.ogrid[0:blue_np.shape[0], 0: blue_np.shape[1]]
    Xball, Yball = raster_ball(blue_np)    

##########  START Hill Climb  ###############
    hill_dict={}
    hill_dict[Xball,Yball]= 1 #max_val[max_val.index(max(max_val))]
    hill_climb(1,0)
    hill_climb(-1,0)
    hill_climb(0,1)
    hill_climb(0,-1)

    if f==0:
        Xball_o = Xball
        Yball_o = Yball
    elif (f_index<3):
        if (math.sqrt( (Xball-Xball_o)**2 + (Xball-Xball_o)**2) > 3):
            f_index = f_index + 1
            print(f_index, f)
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
    print(f, f_index, Xball, Yball, temp_sum, )
    image_data_list.append([f, Xball, Yball, temp_sum,   0,0,0])
print(f_index_list)   
print('image_data =', image_data_list)

########## Plot the pre-hit frame #########
f_number = f_index_list[0][1] -1
file_name = file_list[f_number]    
start_img = Image.open(os.path.join(path_dir, file_name))
start_np = np.array(start_img)
blue_np = start_np[:,:,2]         #blue_img = Image.fromarray(blue_np)
blue_np, back_stats = remove_background(blue_np)

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
    launch_velocity_mps = mm_per_pix*Distance_pix/2
else:
    launch_angle_deg = 0
    launch_velocity_mps =0

print('velocity ={:.2f} m/s'.format(launch_velocity_mps))
print('angle = {:.1f} deg)'.format(launch_angle_deg) ) 
    
ax[4,0].scatter(x_list, y_list)
ax[4,0].set_xlim(0,150)
ax[4,0].set_ylim(-10,65)
ax[4,0].set_xlabel('x pixels')
ax[4,0].set_ylabel('y pixels')
ax[4,0].text(10,50, 'V={:.1f} m/s'.format(launch_velocity_mps) )
ax[4,0].text(10,30, 'A={:.1f} deg'.format(launch_angle_deg) )

plt.show()

#######################################################
######  BACKUP      ##################################
#######################################################

"""
########################################
final_img = Image.fromarray(blue_np)
bbox =  (Xball - Xwidth, Yball - Ywidth, Xball + Xwidth, Yball + Ywidth)
draw = ImageDraw.Draw(final_img)
xxx= draw.ellipse(bbox, outline=255)
#xxx = xxx.rotate(90, expand=1)

fig,ax = plt.subplots(3, figsize=(6,9))
ax[0].imshow(start_img)
#ax[1].imshow(img_blue, cmap='plasma', vmin=(blue_median + 1*blue_std), vmax=150)  #'viridis' 'gist_gray'
ax[1].plot(Xsum)
ax[1].plot(Ysum)
#ax[3].imshow(img[(Xcm - eX):(Xcm + eX), (Ycm - eY/2):(Ycm + eY/2)], cmap='plasma', vmin=50, vmax=80)  #'viridis' 'gist_grayax[3].imshow(img[(Xcm - eX):(Xcm + eX), (Ycm - eY/2):(Ycm + eY/2)], cmap='plasma', vmin=50, vmax=80)  #'viridis' 'gist_gray'
ax[2].imshow(final_img, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'

plt.show()
"""



###### ARRAY Details ##################
#print(imarray)
#[[[55 56 55]
#  [55 56 56]
 # [56 56 56]


#print(imblue)
#[[55 56 56 ... 56 57 57]
# [56 56 56 ... 56 57 57]
# [56 56 55 ... 56 57 57]
# ...
# [56 56 56 ... 58 58 58]
# [55 55 55 ... 57 57 57]
# [55 55 55 ... 57 57 57]]


#######################################################
#xxx = im.histogram()

#plt.imshow(im, cmap='plasma', vmin=50, vmax=80)
#plt.colorbar(location='top')
#plt.colorbar()
#img.show()

#plt.colorbar(location='top')

#ax[1,0].imshow(img, cmap='plasma', vmin=50, vmax=80)  #'viridis' 'gist_gray'
#plt.imshow(im, cmap='plasma', vmin=50, vmax=80)
#plt.colorbar(location='top')

#Remove the Club head
#blue_np[blue_np > 180]=0   #this didn't work?

############################
"""
###########  XY Sum Method  (CM or find_peaks) #######
Xsum = blue_np.sum(axis=0)
temp=[]
for x in np.arange(len(Xsum)):
    temp.append(x*Xsum[x])
Xcm = np.array(temp).sum()/Xsum.sum()
Xpeaks, Xprops = find_peaks(Xsum, width=(20,60), distance=30)
print('Xpeaks =',Xpeaks)
print('Xprops =', Xprops)
#print('Xwidth =', Xprops['widths'][0])

Ysum = blue_np.sum(axis=1)
temp=[]
for y in np.arange(len(Ysum)):
    temp.append(y*Ysum[y])
Ycm = np.array(temp).sum()/Ysum.sum()
print(Xcm, Ycm)
Ypeaks, Yprops = find_peaks(Ysum, width=(30,60), distance=30)
#Yhalfs = peak_widths(Ysum, Ypeaks)

print('Ypeaks = ', Ypeaks)
print('Yprops =', Yprops)
#print('Ywidths = ', Yhalfs)
"""

#eX, eY = 70, 50 #Size of Bounding Box for ellipse
#bbox =  (Xcm - eX/2, Ycm - eY/2, Xcm + eX/2, Ycm + eY/2)
"""
if (Xpeaks.size > 0):
    Xball = Xpeaks[0]
    Xwidth = 30 # Xprops['widths'][0]
    Yball = Ypeaks[0]
    Ywidth = 30 # Yprops['widths'][0]
else:
    Xball = 50
    Yball= 50
    Xwidth=25
    Ywidth=25
"""


'''    
###...Remove the foreground LEDs
    x_LED_o = 95
    y_LED_o = 135
    delta_LED=30
    if foreground_LED:
        print('subtracting LEDs')
        x_LED = [x_LED_o, x_LED_o+35, x_LED_o+80, x_LED_o+130,x_LED_o+165] #[160, 195, 240, 290, 325]
        y_LED = [y_LED_o,y_LED_o,y_LED_o-5,y_LED_o-10, y_LED_o-10] #[130, 130, 125, 120, 120]
        for i in np.arange(len(x_LED)):
            blue_np[y_LED[i]:y_LED[i]+delta_LED, x_LED[i]:x_LED[i]+delta_LED]=0 #.....Mask
        img_blue = Image.fromarray(blue_np)     
    else:
        pass

###...Remove the "background"""
    blue_mean = blue_np.mean()
    blue_median = np.median(blue_np)
    blue_std = blue_np.std()
    print('{} mean ={:.2f}, median ={:.2f}, std ={:.2f}'.format(f,
            blue_mean, blue_median, blue_std))  #56.1 55.0 6.8
    blue_np[blue_np < (blue_median + 1*blue_std)]=0      #...............................Mask
'''

####################################################
""""
start_img = Image.open(os.path.join(path_dir, file_list[0]))
start_np = np.array(start_img)
for f in np.arange(len(file_list)):
    file_name = file_list[f]    
    test_img = Image.open(os.path.join(path_dir, file_name))
    test_np = np.array(test_img)
    change_np = test_np - start_np
    print('f=',f, test_np.sum(), change_np.sum())
"""
######################################################

#    if temp_sum < 50000:
#        image_data_list.append([f, 0,0,0, Xball, Yball, temp_sum ])
#    else:
#        image_data_list.append([f, Xball, Yball, temp_sum,   0,0,0])
