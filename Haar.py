import numpy as np
import cv2 as cv
import sys
import os

import Process

#https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html

face_path = os.path.join(os.path.curdir, "HaarXML", 'haarcascade_frontalface_default.xml')
eye_path = os.path.join(os.path.curdir, "HaarXML", 'haarcascade_eye.xml')


face_cascade = cv.CascadeClassifier(face_path)
eye_cascade = cv.CascadeClassifier(eye_path)
'''
face_cascade = cv.CascadeClassifier('C:\\Users\\iijim\\Desktop\\PyTracker\\\\HaarXML\\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('C:\\Users\\iijim\\Desktop\\PyTracker\\HaarXML\\haarcascade_eye.xml')
'''

eyes =[]


'''Eyes are only ever in the top half of the face. This function filters out anything 
from the lower half of the face that might be detected -> usually nostrils and mouths '''
def filterEyes(eye_cas_list, face_x, face_y, face_w, face_h):
    eyes = []
    for (ex,ey,ew,eh) in eye_cas_list:

        eye_centre = getEyeCentre((ex,ey,ew,eh), face_x, face_y)
        face_y_centre = face_y + int(face_h/2)
        """        
        fcx = face_x + int(face_w/2)
        fcy = face_y + int(face_h/2)
        print("face xy: " + str(fcx) + ", " + str(fcy))
        print("eye xy: " + str(z[0]) + ", " + str(z[1]))
        """
        
        

        """Rules for excluding a false eye"""
        # If the "eye" is not in the top half of the face
        if (eye_centre[1] > face_y_centre):
            continue
        # If the "eye" width is greater than 45% of the face width
        elif(ew > (face_w * 0.4)):
            continue
        # If the "eye" height is greater than 45% of the face height
        elif (eh > (face_h * 0.4)):
            continue
        
        
        eyes.append((ex,ey,ew,eh))
    
    
    return eyes


def getEyeCentre(eye, fx, fy):
    centre = (int(eye[0] + eye[2]/2) + fx, int(eye[1] + eye[3]/2) + fy)
    return(centre)

def getFaces(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    '''
    for (x,y,w,h) in faces:    
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        eyes = getEyes(x,y,w,h, img, gray) 
    '''
    big_face =[]
    big_face.append
    
    if len(faces) == 1:
        return faces, gray
    elif len(faces) > 1:
        big_face.append(faces[0])
        biggest_face = (0,0,0,0) 
        for f in faces:
            if f[3] > biggest_face[3]:
                biggest_face = f
                big_face[0] = biggest_face
        return big_face, gray
    
    
    return None, gray
    
    #return faces, gray
def getEyes(img, gray, faces, binary_threshold, counter):
    
    
    
    
    keypoints = []
    eye_pos = []
    
    
    eye_counter = 0
    
    for (x,y,w,h) in faces:
        
        '''display box around face'''
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        '''roi = region of interest (i.e. face area)'''
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        '''get 'eyes' - will often get nostrils and mouth '''
        eye_cascade_list = eye_cascade.detectMultiScale(roi_gray)
        
        '''filter out anything that is detected in lower half of face'''
        eye_pos = filterEyes(eye_cascade_list, x,y, w, h)  
        
        
        #print("number of eyes detected: " + str(len(eye_pos)))
        
        
        '''TODO: SELECT ONLY the detected features that are most likely to be the left or right eye'''
        
        eye_number = 1
        '''cycle through list of eyes'''
        for (ex,ey,ew,eh) in eye_pos:
            
            #display box around eyes
           cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
           
           ## Extract the eye section for a greyscale image of a detected face
           mask = roi_gray[ey:ey+eh, ex:ex+ew]
           
           '''
           ################################
           s = 'eye ' + str(eye_counter)
           cv.imshow(s,mask)
           cv.waitKey(0)
           ################################
           '''
           
           """ process eyes and find the position of the pupil"""
           pos = Process.process_eye(mask, binary_threshold, counter, eye_number)
           
           
           #print("keypoint " + str(eye_counter) + ": " + str(blobs[0].pt))
           
           
           '''add the coordinates (keypoints) of the pupil to a list of keypoints'''
           keypoints.append(pos)
           
           #######################
           eye_counter = eye_counter + 1
           #######################
           eye_number = eye_number + 1
           
    return keypoints, eye_pos
