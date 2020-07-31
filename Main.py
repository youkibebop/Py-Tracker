import cv2 as cv
import os

import Haar
import Process
from video import genVideo 
#import test

#https://realpython.com/face-detection-in-python-using-a-webcam/



'''Change this to adjust binary threshold value according to lighting'''
'''use this for test video'''
#binary_threshold = 25
binary_threshold = 44



def extractVideoFrames():
    
    filename = os.path.join(os.path.curdir, "Videos", "face_vid.mp4")
	
    cap = cv.VideoCapture(filename)
    
    if not cap.isOpened():
        print('{} not opened', filename)
        sys.exit(1)
    count = 0
    while(count < 101):
        ret, frame = cap.read()

        if not ret:
            break
        frame_path = os.path.join(os.path.curdir, "Frames", "frame%d.png" % count)

        cv.imwrite(frame_path, frame)
        count = count + 1  
    
    return count
        
'''draw a red circle on pupils'''

       



'''main function for a live webcam stream'''
def live_capture_main():
    counter = 0
    
    video_capture = cv.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        
        counter = counter + 1
        
        faces, gray_img = Haar.getFaces(img)
        
        if faces is not None:
        
            keypoints, eyes = Haar.getEyes(img, gray_img, faces, binary_threshold, counter)    
            
            Process.outline_pupils(img, eyes, faces[0], keypoints)  
        
        cv.imshow('Video', img)
    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    video_capture.release()
    cv.destroyAllWindows()

'''main function for a live webcam stream'''
def video_input_main():
    counter = 0
    num_frames = extractVideoFrames() + 1
    shape = [0,0]
    while True:
        
        if (counter == num_frames - 1):
            break
        
        frame = os.path.join(os.path.curdir, "Frames", "frame%d.png" % counter)

        
        img = cv.imread(frame)
        
        shape[0], shape[1], _ = img.shape
        
        
        counter = counter + 1
        
        
        faces, gray_img = Haar.getFaces(img)
        
        if faces is not None:
        
            keypoints, eyes = Haar.getEyes(img, gray_img, faces, binary_threshold, counter)    
            
            Process.outline_pupils(img, eyes, faces[0], keypoints)   
        
        cv.imshow('Video', img)
        
        path = os.path.join(os.path.curdir, "VideoFrames", "f%d.tif" % counter)
        
        """
        cut_w = int(0.25 * shape[1])
        img2 = img[0:shape[0], cut_w:(shape[1] - cut_w)]
        cv.imwrite(path, img2)
        """
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    #path = os.path.join(os.path.curdir, "VideoFrames")
    #genVideo(path,shape,num_frames )
    
    cv.destroyAllWindows()


'''main function for a static image'''
def static_image_main():
    path = os.path.join(os.path.curdir, "StaticImages", 'face.jpg')
    

    img = cv.imread(path)
    cv.imshow('x',img)
    cv.waitKey(0)
    
    faces, gray_img = Haar.getFaces(img)
    
    keypoints, eyes = Haar.getEyes(img, gray_img, faces, binary_threshold, counter)    
    
    l, r = outline_pupils(img, eyes, faces[0], keypoints)  
    
    
    cv.imshow('Video', img)
    cv.waitKey(0)



print()


'''TOGGLE STATIC OR LIVE WEBCAM MODE'''
#############################################

#video_input_main()
#static_image_main()
print("Press q to exit out of webcam video")
print()
print("Tips:")
print()
print("If the red tracker dot is fixed in the middle of the eye...")
print("increase the int binary_threshold at the top of Main.py by 10")
print()
print("If the red tracker dot is going all over the place...")
print("decrease the int binary_threshold at the top of Main.py by 10")



live_capture_main()

