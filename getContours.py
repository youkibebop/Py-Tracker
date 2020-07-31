# import the necessary packages
#import imutils
import cv2
import numpy as np
import math

def getCon(img):
    
    """Some labels"""
    x = 0
    y = 1
    area = 4
    
    
    img = img.astype('uint8')
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    
    if num_components == 0:
        return (int(img.shape[0]/2) + int(img.shape[1]/2)) 
    
    
    MAX = stats[0][area]
    
    d = 0
    
    if math.isnan(centroids[0][x]):
        d = d + 1
    max_comp_pos = [int(centroids[d][x]), int(centroids[d][y])]
    #print(max_comp_pos)
    
    d = d + 1
    """
    m = np.zeros( ( np.array(img).shape[0], np.array(img).shape[1], 3 ) )
    m[:,:,0] = img
    m[:,:,1] = img
    m[:,:,2] = img
    cv2.circle(m, max_comp_pos, 3, (0,0,255), 4 )
    cv2.imshow("h",m)
    cv2.waitKey(0)
    """
    
    for i in range(d, num_components):
        
        
        if stats[i][x] == 0 and stats[i][y] == 0:
            continue
        
        if stats[i][area] > MAX:
            MAX = stats[i][area]
            max_comp_pos = [int(centroids[i][x]), int(centroids[i][y])]
        
        """
        mask = np.zeros( ( np.array(img).shape[0], np.array(img).shape[1], 3 ) )
        mask[:,:,0] = img
        mask[:,:,1] = img
        mask[:,:,2] = img
        cv2.circle(mask, (int(stats[i][0]), int(stats[i][1])), 3, (0,0,255), 4 )
        cv2.imshow("h",mask)
        cv2.waitKey(0)
        """
        
    return max_comp_pos
    """
    for x in output[2]:
        
        if x[0] == 0 and x[1] == 0:
            continue
        
        if x[4] > MAX:
            MAX = x[4]
    
        mask = np.zeros( ( np.array(img).shape[0], np.array(img).shape[1], 3 ) )
        mask[:,:,0] = img
        mask[:,:,1] = img
        mask[:,:,2] = img
        cv2.circle(mask, (int(x[0]), int(x[1])), 3, (0,0,255), 4 )
        cv2.imshow("h",mask)
        cv2.waitKey(0)
     """   
    

    #return cArray

