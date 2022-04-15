from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import math
from math import sqrt, atan2, pi
#from math import sqrt
#import pandas as pd

#from cannySpin import canny_edge_detector
#from collections import defaultdict

#path_dir='20220402_Blue'
#foreground_LED=False

path_dir='20220403_Golfbay1'
foreground_LED=True

Ball_diameter_mm = 42.0
Ball_diameter_pix = 60
mm_per_pix = Ball_diameter_mm/Ball_diameter_pix
ball_radius = 29   #25

image_index = 1

image_data = [[0, 163, 116, 199037, 0, 0, 0],
              [1, 191, 106, 248908, 0, 0, 0],
              [2, 301, 58, 208227, 0, 0, 0],
              [3, 0, 0, 0, 302, 118, 45517]]

Xball = image_data[image_index][1]
Yball = image_data[image_index][2]
Xwidth=Ywidth= ball_radius
bbox =  (Xball - Xwidth, Yball - Ywidth, Xball + Xwidth, Yball + Ywidth)


file_list = os.listdir(path_dir)
file_list.sort()
file_list_len = len(file_list)
print(file_list)  #['out.000108.raw.tiff', out.000109.raw.tiff'....]
print('arrange = ', np.arange(file_list_len) )

fig,ax = plt.subplots(figsize=(10,5))

#####################################
def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    #grayscaled = compute_grayscale(input_pixels, width, height)
#    grayscaled = input_pixels
    masked_np = compute_masked(input_pixels, width, height)
    
    # Blur it to remove noise
    blurred = compute_blur(masked_np, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep

'''
def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
 #           grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
            grayscale[x, y] = pixel[2]
    return grayscale
'''
def compute_masked(input_pixels, width, height):
    masked = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            if (((x-Xball)**2 + (y-Yball)**2)<ball_radius):
                masked[x,y]=pixel
            else:
                masked[x,y]=0
 #           grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
#            grayscale[x, y] = pixel[2]
    return masked

def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = math.sqrt(magx**2 + magy**2)
                direction[x, y] = math.atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)


######################################

#image_data_list=[]
#for f in np.arange(len(file_list)):
#    file_name = file_list[f]    
#    print(f, file_name)
########..............................
while True:
    file_name = file_list[image_index]  
    start_img = Image.open(os.path.join(path_dir, file_name))
    start_np = np.array(start_img)
    #print(imarray.shape)   #(170, 400, 3)
    blue_np = start_np[:,:,2]

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
    print('mean ={:.2f}, median ={:.2f}, std ={:.2f}'.format(blue_mean, blue_median, blue_std))  #56.1 55.0 6.8
    blue_np[blue_np < (blue_median + 1*blue_std)]=0      #...............................Mask

    final_img = Image.fromarray(blue_np)
    draw = ImageDraw.Draw(final_img)
#    spin_img= final_img.copy()
    for x, y in canny_edge_detector(final_img):
#        print(x,y)
        draw.point((x, y), (255))



    xxx= draw.ellipse(bbox, outline=100)
    
#    spin_img=final_img[(Xball - Xwidth):(Xball + Xwidth), (Yball - Ywidth):(Yball + Ywidth) ]
#    ax.imshow(start_img)
    ax.imshow(final_img, cmap='plasma', vmin=60, vmax=150)  #'viridis' 'gist_gray'

    break
plt.show()

