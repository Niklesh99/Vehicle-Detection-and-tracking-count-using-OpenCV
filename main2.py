import cv2
import numpy as np
from time import sleep

width_min=80     #MIN WIDHT
height_min=80    #min height

offset=6   

pos_line=550    #LINE POSITION

delay= 60        # VIDEO FPS

detect = []
cars= 0           # NO of CARS

	
def takes_center(x, y, w, h):          # FRAME CENTER
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')               #Importing Video
subraction = cv2.bgsegm.createBackgroundSubtractorMOG()                 #Subraction Creation

while True:
    ret , frame1 = cap.read()             # read frames from video
    temp = float(1/delay)                 
    sleep(temp) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)          # converts frame to GREY Scale
    blur = cv2.GaussianBlur(grey,(3,3),5)                   # converts gaussian blur
    img_sub = subraction.apply(blur)                        
    dilate = cv2.dilate(img_sub,np.ones((5,5)))             #apply morphological filter to image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))        #Ellipse morphing image
    detected = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    detected = cv2.morphologyEx (detected, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(detected,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)         #find Contours in frame
    
    cv2.line(frame1, (25, pos_line), (1200, pos_line), (255,127,0), 3)           #draw lines on the frames 
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)          #used to draw rect on the frame ROI
        valid_contour = (w >= width_min) and (h >= height_min)     #rect valid only if it is in the frame
        if not valid_contour:                   #Not valid if it is outside
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        #drawing rectangle on ROI 
        centre = takes_center(x, y, w, h)                       #while crossing center
        detect.append(centre)                                   #Appending detect list
        cv2.circle(frame1, centre, 4, (0, 0,255), -1)

        for (x,y) in detect:                                    #if cars crossing the line 
            if y<(pos_line+offset) and y>(pos_line-offset):
                cars+=1                                             # add Count
                cv2.line(frame1, (25, pos_line), (1200, pos_line), (0,127,255), 3)  #draw line in frame
                detect.remove((x,y))                        # remove rect after crossing the line
                print("car is detected : "+str(cars))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)               #DIsplay Video and COUNT 
    cv2.imshow("Detected",detected)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()