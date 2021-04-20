from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

def compute_pointCloud(locationColour, locationDisparity):
    """
    This function computes a point cloud dataframe that can be plotted. This method and 
    implementation was not original and is borrowed from here: 
    https://shkspr.mobi/blog/2018/04/reconstructing-3d-models-from-the-last-jedi/?fbclid=IwAR1E5vMr7IlZQXPoMjSoJBEWkN9xvkVVi1kg0_1pbjeuT3M_hwwvMXEKb8M
    
    This is similar to the algorithm in the last assignment which also creates a numpy array with columns containing 
    the world coordinates x, y, z and red, green, blue data information. The z column are the values from the disparity map
    or depth.
    
    Argument:
    locationColour -- the location of the colour image in the filesystem
    locationDisparity -- the location of the disparity map in the filesystem
    
    Returns:
    df -- returns the point cloud dataframe
    """
    #read colour image
    colourImg    = Image.open(locationColour)
    colourPixels = colourImg.convert("RGB")
    
    #get rgb info
    colourArray  = np.array(colourPixels.getdata()).reshape((colourImg.height, colourImg.width) + (3,))
    
    #get x y info
    indicesArray = np.moveaxis(np.indices((colourImg.height, colourImg.width)), 0, 2)
    imageArray   = np.dstack((indicesArray, colourArray)).reshape((-1,5))
        
    #get z depth info
    depthImg = Image.open(locationDisparity).convert('L')
    depthArray = np.array(depthImg.getdata())
    
    #merge depth info with point cloud
    pointCloud = np.insert(imageArray, 2, depthArray, axis = 1)

    #drop all the pixels left out from the disparity comparison
    pointCloud = pointCloud[pointCloud[:, 2] != 255]
    
    return pointCloud



#A4 code slightly adjusted
def plot_pointCloud(pc, invert = -1):
    '''
    plots the Nx6 point cloud pc in 3D
    assumes (1,0,0), (0,-1,0), (0,0,1) as basis
    '''
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=invert*pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pc[:, 3:][..., ::-1],
            opacity=0.8
        )
    )])
    plot(fig)