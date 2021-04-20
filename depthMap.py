import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def depth_map(imgL, imgR, numDisparities, blockSize, location, scaleValue = 0.7): 
    """
    This function creates a disparity map using semi-global block matching (SGBM). Given two 
    stereo images and the parameters to SGBM, a depth map is calculated and saved. This method and 
    implementation was not original and is borrowed from here: https://timosam.com/python_opencv_depthimage/ 
    
    The algorithm works by matching similar features between these two stereo images by looking 
    for the same pixels in chunks (block sizes). The goal here is by subtracting the maximum and 
    minimum allowed disparity (i.e. the offset), a way to specify the acceptable range for which 
    pixels can move in the blocks, along the epipolar line to get the number of disparities. 

    Argument:
    imgL -- the first stereo image
    imgR -- the second stereo image
    numDisparities -- the number of offsets allowed between images
    blockSize -- the windowsize for image comparison
    location -- path to save images
    scaleValue -- some larger images need to be scaled lower than 70% to be seen on certain pointclouds
    """
    #resize the images
    imgL = cv2.resize(imgL, (math.floor(imgL.shape[1] * scaleValue), math.floor(imgL.shape[0] * scaleValue)))
    imgR = cv2.resize(imgR, (math.floor(imgR.shape[1] * scaleValue), math.floor(imgR.shape[0] * scaleValue)))
    color_image = imgL
    imgL = imgL[:,:,0]
    imgR = imgR[:,:,0]
    
    #defining the parameters for sgbm, more can be found in the link below:
    #https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html#details
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        #The larger the values are, the smoother the disparity is. 
        #P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels.
        P1=8 * 3 * blockSize ** 2,
        #P2 is the penalty on the disparity change by more than 1 between neighbor pixels. 
        #The algorithm requires P2 > P1 .
        #see stereo_match.cpp sample where some reasonably good P1 and P2 values are shown 
        P2=32 * 3 * blockSize ** 2,
        #Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    #creating the same right matcher as the left matcher
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    #gives us hole-free depth images
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setSigmaColor(1.2)
    
    #computes disparity and converts images to desired format
    disparity_left  = left_matcher.compute(imgL, imgR)
    disparity_right = right_matcher.compute(imgR, imgL)
    disparity_left  = np.int16(disparity_left)
    disparity_right = np.int16(disparity_right)
    filteredImg     = wls_filter.filter(disparity_left, imgL, None, disparity_right)
    
    #normalize depth map
    depth_map = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    depth_map = cv2.bitwise_not(depth_map) # Invert image. Optional depending on stereo pair
    cv2.imwrite(location + "disparity.png",depth_map)
    cv2.imwrite(location + "color.png", color_image)
    
    #show original image
    plt.axis('off')
    plt.imshow(color_image)
    plt.show()
    
    #show depth_map
    plt.axis('off')
    plt.imshow(depth_map, cmap='gray')
    plt.show()