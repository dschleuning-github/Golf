#Import Statements
import cv2 as cv
import numpy as np,sys
from matplotlib import pyplot as plt
import pandas as pd
import math

#---------------------------------------------------------------
#PLOT FUNCTIONS
def plt_style_plot(img1,img2):
    plt.subplot(221), plt.imshow(img1, 'gray')
    plt.subplot(222), plt.imshow(img2,cmap = 'gray')
    plt.show()

def cv_style_plot(img1, img2):
#CV style, resizes two images to screen; 0 end
    title_window= 'figure_DAS'
    cv.namedWindow(title_window)
    img_hconcat = cv.hconcat([img1,img2])
    image_width=img_hconcat.shape[1]
    resize_fx=1000.0/image_width
    print('img_hconcat width={} and fx={}'.format(image_width,resize_fx))
    img_smaller = cv.resize(img_hconcat, None, fx = resize_fx, fy = resize_fx, interpolation = cv.INTER_CUBIC)
    cv.imshow(title_window, img_smaller)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
#---------------------------------------------------------
#General Functions
def timecheck(function):
    e1 = cv.getTickCount()
    function
    e2 = cv.getTickCount()
    t = (e2 - e1)/cv.getTickFrequency()
    print('time to do that operation={} s'.format(t) )
  
def plot_circle(img, c_ctr, c_rad):
    dst=img.copy()
    color_cup=(0,0,255)
    thick_cup=1
    cv.circle(dst,center=c_ctr, radius=c_rad, color=color_cup, thickness=thick_cup)
    return dst

def roi_brute(img,center, radius):
    dst=img.copy()
    height, width, depth = img.shape
    print('roi_brute: height=', height, ' width=', width, ' depth=',depth,
          ' center=',center)
    for y in np.arange(height):
        for x in np.arange(width):
            Rpix = np.sqrt((x-center[0])**2 + (y-center[1])**2)
            if Rpix > radius:
                dst[(y,x)]=[0,0,0]
            #pixel_coord=(int(pts1[pts][1]+y -box/2), int(pts1[pts][0]+x-box/2))
            #NOTE swizle in x,y...pts (x,y)....pixel_coords and image[y,x]
            #img[pixel_coord]=[0,0,255]
    return dst


#============================================================
#7. Canny Edge Detections..........OK
def canny(img):
    img=img[:,:,0]    #only works on 1 color? (2D array?)
    threshold1=10   #100
    threshold2=100  #200
    L2gradient=False   #flag for more accurate measure
    edges = cv.Canny(img,threshold1,threshold2,L2gradient)
    return img, edges

# 9A contours:Getting started
def contour_start(img):
    print('start contour_start')
    dst=img.copy()
    
    #STEP0: convert imate to greyscale
    imgray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
#    imgray=dst[:,:,0]
#    print('imgray.shape=',imgray.shape)

    #STEP1:  Create Binary image--threshold or canny
    binary_method= 'threshold'   #'threshold'     #'canny'
    
    if binary_method=='canny':   #CHOOSE METHOD TO MAKE BINARY IMAGE
        #Step1: Binary image==Canny method
        threshold1=10   #100
        threshold2=100  #200
        L2gradient=False   #flag for more accurate measure    
        binary = cv.Canny(imgray,threshold1,threshold2,L2gradient)
    elif binary_method=='threshold':
        #Alternative......Step1: Binary image==Thresholding method
        ret, binary = cv.threshold(imgray, 135, 255, 0)
#        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #STEP2 Find Contours
    mode_das=cv.RETR_TREE                   #??? Documentation TBD
    method_das=cv.CHAIN_APPROX_SIMPLE       #not all points; square reduced to 4 pts
    contours, hierarchy = cv.findContours(binary, mode_das, method_das)
    grey_level=70
    cv.drawContours(dst, contours, -1, (grey_level,grey_level,grey_level), 1)  
#    cv_style_plot(imgray[450:600,150:300],binary[450:600,150:300])      #edges   [450:600,150:300]
    print('finish contour_start')
    return dst, contours, hierarchy

def ellipse_fit(dst, frame, ctr, contours, hierarchy):
    print('start ellipse_fit')
    num_contours=len(contours)
    print('ellipse_fit: num_contours=', num_contours)
    ellipse_list = []
    for index in np.arange(num_contours):
        cnt = contours[index]
        N=len(cnt)
        if (N > 4):
            ellipse_i = cv.fitEllipse(cnt)
            x,y = tuple([int(j) for j in ellipse_i[0] ])          
            e_x,e_y= tuple([int(j) for j in ellipse_i[1] ])
            angle=int(ellipse_i[2])
            delta_x= x - ctr[0]
            delta_y= y - ctr[1]
            theta = int(np.degrees(np.arctan2(delta_y,delta_x) ) )
            r= int(np.sqrt(delta_x**2 + delta_y**2) )
            area= int(e_x*e_y*np.pi )
            ellipse_r= tuple([(x,y),(e_x,e_y),angle])
            if (area > 1) & (area < 1000):
                cv.ellipse(dst,ellipse_r,(0,255,0),1)
                ellipse_list.append([frame, index,N, area, theta,r, ellipse_r])
    ellipse_df=pd.DataFrame(ellipse_list, columns = ['frame','index','N','area','theta','r','ellipse'])
    ellipse_df = ellipse_df.sort_values(by='theta',ascending=True)
#    print(ellipse_df)
    return dst, ellipse_df

def get_ellipses():
    for frame in np.arange(frame_start,frame_end):
        file='frames/' +str(frame) +'.png'   #    frame=151
        img_pepper = cv.imread(file)[0:800,850:1650,:]   #[300:500,1150:1350,:]
        print('\nframe, img_pepper.shape= ',frame,img_pepper.shape) #img_pepper.shape=(1080, 1920, 3)
        img_pepper_roi = roi_brute(img_pepper,pitcher_ctr,pitcher_rad)
        img_pepper_contour, contours_m, hierarcy_m = contour_start(img_pepper_roi)
        img_pepper_ellipse,temp_df=ellipse_fit(img_pepper_contour, frame, pitcher_ctr, contours_m, hierarcy_m)
        if frame==frame_start:
            ellipse_all_df=temp_df
        else:
            ellipse_all_df = pd.concat([ellipse_all_df,temp_df])
    print('\nellipse_all_df.head() =', ellipse_all_df.head())
    ellipse_all_df.to_pickle("./ellipse_all.pkl")
    return ellipse_all_df
#----END ELLIPSE SECTION--------------------------------------------


#----START_TRACKER-------------------------
def tracker(frame, pep, ellipse_df):
    print('\nstart tracker here')
    peppers_140=[204, 158, 184]
    ellipseA=ellipse_df[(ellipse_df['frame'] == frame)
                             &(ellipse_df['index'] == pep)]

    areaA = ellipseA['area'].values[0]
    thetaA = ellipseA['theta'].values[0]
    rA = ellipseA['r'].values[0]
    e_xA = ellipseA['ellipse'].values[0][1][0]
    e_yA = ellipseA['ellipse'].values[0][1][1]
    e_rA=round( e_xA/e_yA ,1)
    pepA= [[frame, pep,areaA, thetaA, rA, e_xA, e_yA, e_rA]]
    pepA_df=pd.DataFrame(pepA,columns=['frame', 'pep', 'areaA',
                                       'thetaA', 'rA', 'e_xA', 'e_yA', 'e_rA'])
    print('pepA_df=\n',pepA_df)

    meritfunc=[]
    frame_n=frame+1
    print('frame_n=',frame_n)
    df_n=ellipse_df[ellipse_df['frame']==frame_n]
    for i in df_n['index'].values:
        D_r = round(100.0*(df_n[df_n['index']==i]['r'].values[0] - rA)/pitcher_rad,1)
        D_rtheta = round(100*(df_n[df_n['index']==i]['theta'].values[0]
                - thetaA)*np.pi*rA/(180.0*pitcher_rad),1)
        rho=round(np.sqrt(D_r**2 + D_rtheta**2),1)
        D_area = abs(round(100*(df_n[df_n['index']==i]['area'].values[0] - areaA)/areaA,1))
        D_ex = abs(round(100*(df_n[df_n['index']==i]['ellipse'].values[0][1][0] - e_xA)/e_xA,1))
        D_ey = abs(round(100*(df_n[df_n['index']==i]['ellipse'].values[0][1][1] - e_yA)/e_xA,1) )
        e_r=round(df_n[df_n['index']==i]['ellipse'].values[0][1][0]/
        df_n[df_n['index']==i]['ellipse'].values[0][1][1],2)
        m_calc= rho + D_ex + D_ey 
        meritfunc.append([frame_n, i,D_r, D_rtheta,rho, D_area, D_ex, D_ey,e_r,m_calc])
    merit_df=pd.DataFrame(meritfunc, columns = ['f_n','i','D_r',
                            'D_rtheta','rho', 'D_area','D_ex','D_ey','e_r','m_calc' ])
    print('merit_df=\n',merit_df[merit_df['f_n']==frame_n].sort_values('m_calc',ascending=True))
    best_index = merit_df[merit_df['f_n']==frame_n].sort_values('m_calc',ascending=True).values[0][1]
    print('best next object is...',best_index)
    return best_index

def initial_tracking_plot(ellipse_all_df):
    #defined in Global constants:   frame_start=140
    
    file0='frames/' +str(frame_start) +'.png'   #    frame=151
    img0 = cv.imread(file0)[0:800,850:1650,:]   #[300:500,1150:1350,:]

    f140_i204_df=ellipse_all_df[(ellipse_all_df['frame']==frame_start)
                                & (ellipse_all_df['index']==204) ]
    print('f140_i204_df =\n',f140_i204_df)


    start_frame_df=ellipse_all_df[ellipse_all_df['frame']==frame_start].sort_values('area',ascending=False)
    start_df=start_frame_df[start_frame_df['area'] <600.0].head(20)
    print('start_df=\n',start_df)
    frame_ellipses = start_df['ellipse']
    print('starting elipses=\n', frame_ellipses)
    
    for e in frame_ellipses:
        print('e=', e)
        cv.ellipse(img0,e,(0,255,0),1)               
    img0= plot_circle(img0,pitcher_ctr,pitcher_dot)
    img0= plot_circle(img0,pitcher_ctr,pitcher_rad)
    cv.putText(img0,str(frame_start),(300,100), font, 2,(255,255,255),1,cv.LINE_AA)

#            best_index=tracking_df[tracking_df['frame']==frames[i]].values[0]
#            current_frame=int(best_index[0])
#            best_index=int(best_index[1])
#            cv.putText(images[i],str(best_index),(300,150), font, 1,(255,255,255),1,cv.LINE_AA)
#            print(current_frame, best_index)
#            ellipse_best_df=ellipse_all_df[(ellipse_all_df['frame'] == current_frame )
#                 &(ellipse_all_df['index'] == best_index)]   #A_141, 141, 238
#            ellipse_best=ellipse_best_df['ellipse'].values[0]
#        print(coords_best_index['ellipse'].values[0])
#            cv.ellipse(images[i], ellipse_best,(0,0,255),1)

    title_window= 'figure_DAS'
    cv.namedWindow(title_window)
#        img_hconcat = cv.hconcat([images[0],images[1],images[2]])
    image_width=img0.shape[1]
    resize_fx=700.0/image_width
    img_smaller = cv.resize(img0, None, fx = resize_fx, fy = resize_fx, interpolation = cv.INTER_CUBIC)
    cv.imshow(title_window, img_smaller)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_tracking_list(ellipse_all_df):
    initial_tracking_plot(ellipse_all_df)

    small_df= ellipse_all_df[(ellipse_all_df['theta'] > theta_start)
                  &(ellipse_all_df['theta'] < theta_stop)]      # & (ellipse_all_df['frame'] == 141 )

    tracking_list=[[140, 204]]    
    print(ellipse_all_df['frame'].unique())

#    for f in ellipse_all_df['frame'].unique():
    frame_list=np.arange(frame_start,frame_end)
    for f in frame_list:
        if f ==frame_start:
            pass
        else:
            n_index = tracker(tracking_list[-1][0],tracking_list[-1][1], small_df)
            tracking_list.append([f,n_index])
    tracking_df=pd.DataFrame(tracking_list, columns=['frame', 'index'])
    print('tracking_df=\n', tracking_df)
    tracking_df.to_pickle("./tracking_df.pkl")
    return(tracking_df)


def plot_tracking(ellipse_all_df, tracking_df):
    frame_c=1
    while frame_c != 0:
        print('\n start plot_tracks here')
        frame_c = int(input('center frame?'))
        print('I got ',frame_c)
    
        file0='frames/' +str(140) +'.png'   #    frame=151
        img0 = cv.imread(file0)[0:800,850:1650,:]   #[300:500,1150:1350,:]
        file1='frames/' +str(frame_c) +'.png'   #    frame=151
        img1 = cv.imread(file1)[0:800,850:1650,:]   #[300:500,1150:1350,:]
        file2='frames/' +str(frame_c+1) +'.png'   #    frame=151
        img2 = cv.imread(file2)[0:800,850:1650,:]   #[300:500,1150:1350,:]

        frames=[140, frame_c, frame_c+1]
        images=[img0,img1,img2]
#    colors=[(255,0,0),(0,255,0),(255,255,255)]
#        for i in np.arange(len(images)):
#            print(i)
#            frame_ellipses = ellipse_all_df[ellipse_all_df['frame']==frames[i]]['ellipse']
#            for e in frame_ellipses:
#                cv.ellipse(images[i],e,(0,255,0),1)  #green for all three.
#                if i==0:
#                    cv.ellipse(images[i+1],e,(255,0,0),1)
#                elif i==2:
#                    cv.ellipse(images[i-1],e,(255,255,255),1)                
#            images[i]= plot_circle(images[i],pitcher_ctr,pitcher_dot)
#            images[i]= plot_circle(images[i],pitcher_ctr,pitcher_rad)
#            cv.putText(images[i],str(frames[i]),(300,100), font, 2,(255,255,255),1,cv.LINE_AA)

#            best_index=tracking_df[tracking_df['frame']==frames[i]].values[0]
#            current_frame=int(best_index[0])
#            best_index=int(best_index[1])
#            cv.putText(images[i],str(best_index),(300,150), font, 1,(255,255,255),1,cv.LINE_AA)
#            print(current_frame, best_index)
#            ellipse_best_df=ellipse_all_df[(ellipse_all_df['frame'] == current_frame )
#                 &(ellipse_all_df['index'] == best_index)]   #A_141, 141, 238
#            ellipse_best=ellipse_best_df['ellipse'].values[0]
#        print(coords_best_index['ellipse'].values[0])
#            cv.ellipse(images[i], ellipse_best,(0,0,255),1)

        #frame0 elipses
        frame_ellipses = ellipse_all_df[ellipse_all_df['frame']==frames[0]]['ellipse']
        for e in frame_ellipses:
            cv.ellipse(images[0],e,(255,0,0),1)  #blue for first frame
            cv.ellipse(images[1],e,(255,0,0),1)  #blue for middle frame

        frame_ellipses = ellipse_all_df[ellipse_all_df['frame']==frames[1]]['ellipse']
        for e in frame_ellipses:
            cv.ellipse(images[1],e,(0,255,0),1)  #green for middle frame

        frame_ellipses = ellipse_all_df[ellipse_all_df['frame']==frames[2]]['ellipse']
        for e in frame_ellipses:
            cv.ellipse(images[2],e,(255,255,255),1)  #white for last frame
            cv.ellipse(images[1],e,(255,255,255),1)  #white for middle frame
             
        for i in np.arange(len(images)):
            print(i)
            images[i]= plot_circle(images[i],pitcher_ctr,pitcher_dot)
            images[i]= plot_circle(images[i],pitcher_ctr,pitcher_rad)
            cv.putText(images[i],str(frames[i]),(300,100), font, 2,(255,255,255),1,cv.LINE_AA)
            best_index=tracking_df[tracking_df['frame']==frames[i]].values[0]
            current_frame=int(best_index[0])
            best_index=int(best_index[1])
            cv.putText(images[i],str(best_index),(300,150), font, 1,(255,255,255),1,cv.LINE_AA)
#            print(current_frame, best_index)
            ellipse_best_df=ellipse_all_df[(ellipse_all_df['frame'] == current_frame )
                 &(ellipse_all_df['index'] == best_index)]   #A_141, 141, 238
            ellipse_best=ellipse_best_df['ellipse'].values[0]
#        print(coords_best_index['ellipse'].values[0])
            cv.ellipse(images[i], ellipse_best,(0,0,255),1)

        print('tracking_df=\n',tracking_df)
        pd_indices=tracking_df.index[tracking_df['frame'] <= frame_c].to_list()
        print(pd_indices)
        for i in pd_indices:
            row_i=tracking_df.iloc[i].values
            frame_i=int(row_i[0])
            index_i=int(row_i[1])
            object_i=ellipse_all_df[(ellipse_all_df['frame']==frame_i) &
                                (ellipse_all_df['index']==index_i)].values[0]
            ellipse_i=object_i[6]
            x_y_i=ellipse_i[0]
            x_y_int=(int(x_y_i[0]), int(x_y_i[1]) )
            print(i, frame_i, index_i, x_y_i, x_y_int)
            images[1]= plot_circle(images[1],x_y_int,2)          
            if i!=0:
                print(x_y_before,x_y_int)
                cv.line(images[1],x_y_before,x_y_int,(255,0,0),1)
                x_y_before=x_y_int
            else:
                x_y_before=x_y_int
            
        title_window= 'figure_DAS'
        cv.namedWindow(title_window)
        img_hconcat = cv.hconcat([images[0],images[1],images[2]])
        image_width=img_hconcat.shape[1]
        resize_fx=1500.0/image_width
        img_smaller = cv.resize(img_hconcat, None, fx = resize_fx, fy = resize_fx, interpolation = cv.INTER_CUBIC)
        cv.imshow(title_window, img_smaller)
        cv.waitKey(0)
        cv.destroyAllWindows()

def contour_main():
#    ellipse_all_df=get_ellipses()
    ellipse_all_df = pd.read_pickle("./ellipse_all.pkl")

#    tracking_df=get_tracking_list(ellipse_all_df)
    tracking_df=pd.read_pickle("./tracking_df.pkl")
    
#    plot_ellipses(141,ellipse_all_df)
    plot_tracking(ellipse_all_df,tracking_df)

#-------------------------------------------------------
#GLOBAL CONSTANTS
font = cv.FONT_HERSHEY_SIMPLEX

frame_start=140
frame_end=155    #180

theta_start = -100
theta_stop = 180

delta_t= .033               #seconds;   Pixel3 slow motion= 30 frames/sec (verified 8/2/2020) 
pitcher_ctr=(320,420)       #Full   (220,520) #Small Circle
pitcher_rad =300            #Full   50 #small circle
pitcher_dot =5

contour_main()



#-----------------------------------------------------
#      OLD
#-----------------------------------------------------

