from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

#path_dir='20220402_Blue\Blue1'
#file_name = 'out.000001.raw.tiff'
path_dir='20220403_Golfbay1'
file_list = os.listdir(path_dir)
#print(file_list)  #['out.000108.raw.tiff',
#'out.000109.raw.tiff', 'out.000110.raw.tiff', 'out.000111.raw.tiff']

file_name = file_list[0]

###.....open .tiff file
im = Image.open(os.path.join(path_dir, file_name))

###......create numpy array
imarray = np.array(im)
#print(imarray.shape)   #(170, 400, 3)
#print(imarray)
#[[[55 56 55]
#  [55 56 56]
 # [56 56 56]

blue_np = imarray[:,:,2]
#print(imblue)
#[[55 56 56 ... 56 57 57]
# [56 56 56 ... 56 57 57]
# [56 56 55 ... 56 57 57]
# ...
# [56 56 56 ... 58 58 58]
# [55 55 55 ... 57 57 57]
# [55 55 55 ... 57 57 57]]

###...Remove the foreground LEDs
x_LED = [160, 195, 240, 290, 325]
y_LED = [130, 130, 125, 120, 120]
delta_LED=20
for i in np.arange(len(x_LED)):
    blue_np[y_LED[i]:y_LED[i]+delta_LED, x_LED[i]:x_LED[i]+delta_LED]=0
img_blue = Image.fromarray(blue_np)

###...Remove the "background"""
blue_mean = blue_np.mean()
blue_median = np.median(blue_np)
blue_std = blue_np.std()
print(blue_mean, blue_median, blue_std)  #57.93914705882353 58.0 11.878032238579193
blue_np[blue_np < (blue_median + 1*blue_std)]=0

#Remove the Club head
#blue_np[blue_np > 180]=0   #this didn't work?

Xsum = blue_np.sum(axis=0)
temp=[]
for x in np.arange(len(Xsum)):
    temp.append(x*Xsum[x])
Xcm = np.array(temp).sum()/Xsum.sum()

Ysum = blue_np.sum(axis=1)
temp=[]
for y in np.arange(len(Ysum)):
    temp.append(y*Ysum[y])
Ycm = np.array(temp).sum()/Ysum.sum()
print(Xcm, Ycm)
Ypeaks,_ = find_peaks(Ysum)
Yhalfs = peak_widths(Ysum, Ypeaks)

print('peaks = ', Ypeaks, 'widths = ', Yhalfs[0])

img = Image.fromarray(blue_np)

eX, eY = 70, 70 #Size of Bounding Box for ellipse
bbox =  (Xcm - eX/2, Ycm - eY/2, Xcm + eX/2, Ycm + eY/2)
draw = ImageDraw.Draw(img)
draw.ellipse(bbox, outline=255)

fig,ax = plt.subplots(4, figsize=(6,9))
ax[0].imshow(im)
ax[1].imshow(img_blue, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'
ax[2].plot(Xsum)
ax[2].plot(Ysum)
#ax[3].imshow(img[(Xcm - eX):(Xcm + eX), (Ycm - eY/2):(Ycm + eY/2)], cmap='plasma', vmin=50, vmax=80)  #'viridis' 'gist_grayax[3].imshow(img[(Xcm - eX):(Xcm + eX), (Ycm - eY/2):(Ycm + eY/2)], cmap='plasma', vmin=50, vmax=80)  #'viridis' 'gist_gray'
ax[3].imshow(img, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'

plt.show()

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
