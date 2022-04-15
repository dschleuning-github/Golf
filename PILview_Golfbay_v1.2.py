from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import math
#from math import sqrt
#import pandas as pd

########### Set the Path ########
#path_dir='20220402_Blue'
#foreground_LED=False

path_dir='20220403_Golfbay1'
foreground_LED=True

############ Some Global constants ##########
Ball_diameter_mm = 42.0
Ball_diameter_pix = 60
mm_per_pix = Ball_diameter_mm/Ball_diameter_pix
ball_radius = 29   #25




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
#print(file_list)  #['out.000108.raw.tiff', out.000109.raw.tiff'....]
#print('arrange = ', np.arange(file_list_len) )

fig,ax = plt.subplots(file_list_len+1,2, figsize=(8,7))

image_data_list=[]
for f in np.arange(len(file_list)):
    file_name = file_list[f]    
#    print(f, file_name)
########..............................

    start_img = Image.open(os.path.join(path_dir, file_name))
    start_np = np.array(start_img)
#print(imarray.shape)   #(170, 400, 3)
    blue_np = start_np[:,:,2]
#blue_img = Image.fromarray(blue_np)

###...Remove the foreground LEDs
    if foreground_LED:
        x_LED = [160, 195, 240, 290, 325]
        y_LED = [130, 130, 125, 120, 120]
        delta_LED=20
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



######### Raster using radius mask ########
    Xsum = blue_np.sum(axis=0)
    Ysum = blue_np.sum(axis=1)
#    print('len(Ysum)=', len(Xsum))

    y_temp,x_temp = np.ogrid[0:len(Ysum), 0: len(Xsum)]
    ball_dict={}
    for x in np.arange(0,len(Xsum),10):
        for y in np.arange(0,len(Ysum),10):
            temp_np = blue_np.copy()
            mask = (x-x_temp)**2+(y-y_temp)**2 <= ball_radius**2
            temp_sum = temp_np[mask].sum()
            ball_dict[x,y] = temp_sum
    max_val = list(ball_dict.values())
    max_ke = list(ball_dict.keys())
    Xball, Yball = max_ke[max_val.index(max(max_val))]
#    print('raster =', Xball, Yball, ball_radius)

#...START Hill Climb
    hill_dict={}
    hill_dict[Xball,Yball]=max_val[max_val.index(max(max_val))]
    hill_climb(1,0)
    hill_climb(-1,0)
    hill_climb(0,1)
    hill_climb(0,-1)
#    print('hill_dict =', hill_dict)
#    print('Xball= {}, Yball={}'.format(Xball, Yball))
    Xwidth=Ywidth=ball_radius

    final_img = Image.fromarray(blue_np)
    bbox =  (Xball - Xwidth, Yball - Ywidth, Xball + Xwidth, Yball + Ywidth)
    draw = ImageDraw.Draw(final_img)
    xxx= draw.ellipse(bbox, outline=255)
    ax[f,0].imshow(start_img)
    ax[f,1].imshow(final_img, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'

    if temp_sum < 100000:
        image_data_list.append([f, 0,0,0, Xball, Yball, temp_sum ])
    else:
        image_data_list.append([f, Xball, Yball, temp_sum,   0,0,0])
    
#print('image_data =', image_data_list)

time_list=[]
x0 = image_data_list[0][1]
y0 = image_data_list[0][2]
x_list = []
y_list = []
T_pulse = 2 #mX
for f in np.arange(len(file_list)):
    if(image_data_list[f][3] != 0):
        time_list.append(image_data_list[f][0]*T_pulse)
        x_list.append(image_data_list[f][1] - x0)
        y_list.append(-(image_data_list[f][2] - y0))

launch_angle_deg =  180/math.pi*math.atan2((y_list[2] - y_list[1]),(x_list[2] - x_list[1]) )
Distance_pix = math.sqrt( (y_list[2] - y_list[1])**2 + (x_list[2] - x_list[1])**2)
launch_velocity_mps = mm_per_pix*Distance_pix/2

print('velocity ={:.2f} m/s'.format(launch_velocity_mps))
print('angle = {:.1f} deg)'.format(launch_angle_deg) ) 
    
ax[f+1,0].scatter(x_list, y_list)
ax[f+1,0].set_xlim(0,300)
ax[f+1,0].set_ylim(-20,120)
ax[f+1,0].set_xlabel('x pixels')
ax[f+1,0].set_ylabel('y pixels')
ax[f+1,0].text(10,100, 'V={:.1f} m/s'.format(launch_velocity_mps) )
ax[f+1,0].text(10,80, 'A={:.1f} deg'.format(launch_angle_deg) )

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
