import open3d as o3d
import numpy as np


def plot_mesh(pc, invert=True, color="height"):
    """
    esimmates normals for a point cloud and reconstructs a
    triangle mesh using poisson surface reconstruction
    """
    #inverts the z axis if requested
    if invert:
        pc[:, 2] = -pc[:, 2]  
    #changes the points into a numpy array and normalizes them
    #with respect to the largest value  
    points = np.array([pc[:, 1], -pc[:, 0], pc[:, 2]]).transpose()
    points = points / np.max(points)
    
    #creates an open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    #estimates the normals of the point cloud
    pcd.estimate_normals()
    
    #invert normals
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    
    #visualize the point cloud
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    #reconstructs and colors a triangle mesh that represents the point cloud
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #reconstructs the mesh
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        verts = np.asarray(mesh.vertices)
        #colors the mesh
        if color == "random":
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.random.rand(verts.shape[0], 3).astype(np.float64))
        elif color == "height":
            mesh.vertex_colors = o3d.utility.Vector3dVector((verts**2 / np.max(verts)).astype(np.float64))
    
    mesh.compute_triangle_normals()
    #visualize the mesh
    o3d.visualization.draw_geometries([mesh])