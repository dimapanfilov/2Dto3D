import cv2
from pointCloud import compute_pointCloud, plot_pointCloud
from plotMesh import plot_mesh
from depthMap import depth_map

#read images
imgL1 = cv2.imread('ex1/left.png',cv2.COLOR_RGB2GRAY)[..., ::-1]
imgR1 = cv2.imread('ex1/right.png',cv2.COLOR_RGB2GRAY)[..., ::-1]

imgL2 = cv2.imread('ex2/left.jpg',cv2.COLOR_RGB2GRAY)[..., ::-1]
imgR2 = cv2.imread('ex2/right.jpg',cv2.COLOR_RGB2GRAY)[..., ::-1]

imgL3 = cv2.imread('ex3/left.png',cv2.COLOR_RGB2GRAY)[..., ::-1]
imgR3 = cv2.imread('ex3/right.png',cv2.COLOR_RGB2GRAY)[..., ::-1]

imgL4 = cv2.imread('ex4/left.png',cv2.COLOR_RGB2GRAY)[..., ::-1]
imgR4 = cv2.imread('ex4/right.png',cv2.COLOR_RGB2GRAY)[..., ::-1]

imgL5 = cv2.imread('ex5/left.png',cv2.COLOR_RGB2GRAY)[..., ::-1]
imgR5 = cv2.imread('ex5/right.png',cv2.COLOR_RGB2GRAY)[..., ::-1]

#img1 pointcloud
depth_map(imgL1, imgR1, 16, 5, 'ex1/')
pc = compute_pointCloud('ex1/color.png', 'ex1/disparity.png')
plot_pointCloud(pc)
plot_mesh(pc)

# #img2 pointcloud
depth_map(imgL2, imgR2, 100, 11, 'ex2/')    #I tested multiple hyperparameters and think this is the best
pc = compute_pointCloud('ex2/color.png', 'ex2/disparity.png')
plot_pointCloud(pc)
plot_mesh(pc)


#img3 pointcloud
depth_map(imgL3, imgR3, 128, 11, 'ex3/', 0.6)
pc = compute_pointCloud('ex3/color.png', 'ex3/disparity.png')
plot_pointCloud(pc, 1)
plot_mesh(pc, invert=False)

#img4 pointcloud
depth_map(imgL4, imgR4, 16, 11, 'ex4/')
pc = compute_pointCloud('ex4/color.png', 'ex4/disparity.png')
plot_pointCloud(pc)
plot_mesh(pc)

#img5 pointcloud
depth_map(imgL5, imgR5, 32, 3, 'ex5/', 0.5)
pc = compute_pointCloud('ex5/color.png', 'ex5/disparity.png')
plot_pointCloud(pc)
plot_mesh(pc)
