#!/usr/bin/env python3
#Schleuning Maker software

import time
from picamera import PiCamera
from time import sleep
import os
#from rpi_ws281x import *
#import argparse
import RPi.GPIO as GPIO
#from Dadclass import hbridge
#import random
#import pygame
import sys
sys.path.append("..")
sys.path.append("/usr/bin")
sys.path.append('/home/pi/.local/lib/python3.7/site-packages')
#import vlc

####################################################3
#...........GPIO SETUP.....................
####################################################
GPIO.setmode(GPIO.BOARD) #Set GPIO to pin numbering
GPIO.setwarnings(False)

##################################################
#  setup motion sensor
##################################################
led = 16 #23 #Assign pin 10 to LED
pir = 32 #GPIO11 #Assign  to PIR
GPIO.setup(led, GPIO.OUT) #Setup GPIO pin for LED as output
GPIO.setup(pir, GPIO.IN) #Setup GPIO pin PIR as input

############################################
# Setup the camera
#########################
#camera = PiCamera()
#camera.rotation = 180

########################################################
# MAIN CODE STARTS HERE
######################################################
try:
    while True:
        if GPIO.input(pir) == True: #If PIR pin goes high, motion is detected
            print ("Motion Detected!")
            GPIO.output(led, True) #Turn on LED
#           camera.start_preview()
#            camera.start_recording('/home/pi/Desktop/video.h264')
#            sleep(1)
#            camera.stop_recording()
#            camera.stop_preview()

#            os.system("echo is this like a print statement")
#            for i in range(5):
#                os.system("raspistill -o Desktop/image%.jpg -t 1" % i)
            os.system("raspistill -o image.jpg -t 1")
#           raspistill -o ~/Desktop/image0.jpg")
#            os.system("raspistill -o ~/Desktop/image1.jpg -t 1 -n")
            time.sleep(0.1)
            print("  ")
            break
        else:
#            print ("waiting....")
            GPIO.output(led, False) #Turn on LED

except KeyboardInterrupt: #Ctrl+c
    pass #Do nothing, continue to finally

finally:
    GPIO.output(led, False) #Turn off LED in case left on
    GPIO.cleanup() #reset all GPIO
    print ("Program ended")
#print("The skit we did was skit", Random_Number)
print("all done")


############################
#.....define the right_arm object................
#freq = 500   #Hz:  GPIO.PWM(channel, frequency) 50Hz => 20mS;
#right_in1 = 31
#right_in2 = 33
#right_en = 35
#right_arm = hbridge('right', right_in1, right_in2, right_en, freq)
#right_arm.say_hi()
#Random_Number = random.randint(0, 1)