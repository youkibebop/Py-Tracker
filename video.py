import cv2 as cv
import os

def genVideo(path, shape, num_frames):
    h = shape[0]
    w = shape[1]
    vid = cv.VideoWriter('project.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, (w,h))
    for i in range (1, num_frames ):
        frame_path = os.path.join(path, "f%d.tif" % i)
        frame = cv.imread(frame_path)
        vid.write(frame)
    cv.destroyAllWindows()
    vid.release()



"""    
path = os.path.join(os.path.curdir, "VideoFrames", "f1.tif")
i = cv.imread(path)
print(i.shape)
shape = [0,0]
shape[0] = 1080
shape[1] = 960    
path = os.path.join(os.path.curdir, "VideoFrames")
genVideo(path,shape,100 )
"""