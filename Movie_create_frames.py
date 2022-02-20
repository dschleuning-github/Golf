import numpy as np
import cv2
import os

##################################
####ADD your file here############
#cap = cv2.VideoCapture('RGB_WS2811_test2.mp4')
#path_output_dir='test2'

#cap = cv2.VideoCapture('sony_test2.mp4')
#path_output_dir='sony_test2'

#cap = cv2.VideoCapture('2022_01_27 Daniel Driver.mp4')
#path_output_dir='2022_01_27 Daniel Driver'

#cap = cv2.VideoCapture('2022_01_27 Daniel Driver.mp4')
#path_output_dir='2022_01_27 Daniel Driver'

cap = cv2.VideoCapture("2022_02_06_golfBay/Max7iron/Max7ironBall.MP4")
path_output_dir='2022_02_06_golfBay/Max7iron/Ball'

#print(cap.shape)

#################################

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.
if int(major_ver)  < 3 :
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  print('Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}'.format(fps))
else :
  fps = cap.get(cv2.CAP_PROP_FPS)
  print('Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}'.format(fps))

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file") 
# Read until video is completed

count = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, frame)
        count += 1
        print(count)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25)== ord('q'):
            print('Ok you want to stop...close everthing')
            break
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
print('everything closed')
