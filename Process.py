import cv2 as cv
import os
import numpy as np

from getContours import getCon

def writeEye(filename, img):
    base_path = os.path.join(os.path.curdir, "Eyes")
    path = os.path.join(base_path, filename)
    cv.imwrite(path, img)   

def outline_pupils(img, eyes, face, keypoints):
    right_pos = [-1,-1]
    left_pos = [-1,-1]
    
    if (len(face) > 0 and len(eyes) > 0):
        
        w = face[2]
        
        left_est = (int(w * 0.1), int(w * 0.45))
        right_est = (int(w * 0.55), int(w * 0.9))
        
        for q in range(len(eyes)):
            
            eye_mid = eyes[q][0] + int(eyes[q][2]/2)
            
            '''x position of each eye'''
            x = face[0] + eyes[q][0]
            
            '''y position of each eye'''
            y = face[1] + eyes[q][1] #+ int(0.25 * eyes[q][1])    
            
            if left_est[0] < eye_mid and eye_mid < left_est[1]:
                if (len(keypoints[q]) != 0):
                    left_pos = (int(keypoints[q][0]) + x, int(keypoints[q][1]) + y)
            elif right_est[0] < eye_mid and eye_mid < right_est[1]:
                if (len(keypoints[q]) != 0):
                    right_pos = (int(keypoints[q][0]) + x, int(keypoints[q][1]) + y)
            
            '''if a centroid for the pupil could be found, draw a circle of it onto the image'''
        if right_pos[0] != -1:
            cv.circle(img, (right_pos[0], right_pos[1]), 3, (0,0,255), 4 )
        if left_pos[0] != -1:
            cv.circle(img, (left_pos[0], left_pos[1]), 3, (0,0,255), 4 )
       
        """
        if (len(keypoints[q]) != 0):
            cv.circle(img, (int(keypoints[q][0]) + x, int(keypoints[q][1]) + y), 3, (0,0,255), 4 )
        """
        
    return left_pos, right_pos

def process_eye(img, binary_threshold, counter, eye_num):    
    
    #only save the 100 most recent images
    count_mod = counter % 100
    
    #save a greyscale image of captured eye
    filename = "step_1_eye%d_%d.png" % (eye_num, count_mod)
    writeEye(filename, img)
    
    
    
    #reduce the grey image to a binary image with a binary threshold
    ret, mask = cv.threshold(img, binary_threshold, 255, cv.THRESH_BINARY)
    
    

    #save binary image of eye
    filename = "step_2_bineye%d_%d.png" % (eye_num, count_mod)
    writeEye(filename, mask)
    
    
    #remove eyebrows from the processed image
    mask = remove_eyebrows(mask)
    
    #save processed image without eyebrows
    filename = "step_4_nobrow%d_%d.png" % (eye_num, count_mod)
    writeEye(filename, mask)
    
###################################################################################################    
    '''Closing operation to reduce the binary images to fewer and larger connected components'''
    mask = cv.erode(mask, None, iterations=3)
    mask = cv.dilate(mask, None, iterations=5) 
    mask = cv.medianBlur(mask, 5)
####################################################################################################
    
    #save processed image of eye
    filename = "step_3_proceye%d_%d.png" % (eye_num, count_mod)
    writeEye(filename, mask)
    

    
    #blur the image once more for keypoint detection (detecting 'blobs')
    mask = cv.medianBlur(mask, 5)
    
    
    
    img2 = np.zeros( ( np.array(img).shape[0], np.array(img).shape[1], 3 ) )
    img2[:,:,0] = img # same value in each channel
    img2[:,:,1] = img
    img2[:,:,2] = img
    mask2 = np.zeros( ( np.array(mask).shape[0], np.array(mask).shape[1], 3 ) )
    mask2[:,:,0] = mask # same value in each channel
    mask2[:,:,1] = mask
    mask2[:,:,2] = mask
    
    """
    #find blobs using blob-detector we passed in the function input params
    blobs = detector.detect(mask)

    #fuction for drawing coordinates of blobs (NOT working atm)
    if (len(blobs) != 0):
        cv.circle(mask2, (int(blobs[0].pt[0]), int(blobs[0].pt[1])), 3, (0,0,255), 4 )
        cv.circle(img2, (int(blobs[0].pt[0]), int(blobs[0].pt[1]) + int(0.25 * img.shape[1])), 3, (0,0,255), 4 )
    """
    
    pos = getCon(mask)
    
    
    cv.circle(mask2, (pos[0], pos[1]), 3, (0,0,255), 4 )
    
    
    pos[1] = pos[1] + int(0.25 * img.shape[1])
    
    cv.circle(img2, (pos[0], pos[1]), 3, (0,0,255), 4 )
    
    
    
    filename = "step_5_keypoints%d_%d.png" % (eye_num, count_mod)
    writeEye(filename, mask2) 
    filename = "step_6_pupil%d_%d.png" % (eye_num, count_mod)
    writeEye(filename, img2) 
    
    return pos


def remove_eyebrows(img):
    height, width = img.shape
    cut_h = int(0.25 * height)
    new_img = img[cut_h:height, 0:width] 

    return new_img



