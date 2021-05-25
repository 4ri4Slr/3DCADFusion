import utils
import time
import cv2
import numpy as np
import random
import fusion
import trimesh
import predict_mask
import os
if __name__ == "__main__":


    path = './data/gripper'
    res = [1500, 1800]
    d_t = 0.93

    #predict_mask.predict(path, res)
    utils.execute_registeration_sequence(path, d_t, res)
    #utils.execute_depth_generation_sequence( path , res = [1600,1200])
    print("Estimating voxel volume bounds...")
    n_imgs = 100
    cam_intr = np.loadtxt(path + "/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    cam_pose = np.loadtxt(path + "/frame-%05d.pose.txt" % 0)  # 4x4 rigid transformation matrix
    for i in range(n_imgs):
        #i = i+17 # Read depth image and camera pose
        #depth_im = cv2.imread(path + "/frame-%05d.depth.png" % (i + 1), -1).astype(float)
        depth_im = np.load( path + '/xyz/%d.npy' % i)
        mask = cv2.imread(path + '/masks/frame-' + str(i).zfill(6) + '.color.jpg')
        depth_im = utils.clean_depth(depth_im[:,:,2], mask, res)
        print( path + "/frame-%05d.depth.png" % (i + 1))
        # depth_im = depth_im / 255
        # depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt(path + "/frame-%05d.pose.txt" % (i))
        cam_pose = np.linalg.inv(cam_pose)
        print( path + "/frame-%05d.pose.txt" % (i))
        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.002)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))


        #i= i+17
        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(path + "/rectified_rgb/image%05d.png" % (i + 1)), cv2.COLOR_BGR2RGB)
        #depth_im = cv2.imread( path + "/frame-%05d.depth.png" % (i + 1), -1).astype(float)
        #depth_im = depth_im / 255
        depth_im1 = np.load( path + '/xyz/%d.npy' % i)
        mask = cv2.imread(path + '/masks/frame-' + str(i).zfill(6) + '.color.jpg')
        depth_im = utils.clean_depth(depth_im1[:, :, 2], mask, res)
        cam_pose = np.loadtxt(path + "/frame-%05d.pose.txt" % (i))  # 4x4 rigid transformation matrix

        # Integrate observation into voxel volume (assume color aligned with depth)
        #if i ==0:
        cam_pose = np.linalg.inv(cam_pose)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        # else:
        #     verts, faces, norms, colors = tsdf_vol.get_mesh()
        #     cam_pose = utils.register_tsdf(verts, colors, depth_im1, color_image, mask, cam_intr, cam_pose , res)
        #     cam_pose = np.linalg.inv(cam_pose)
        #     tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    mesh = trimesh.load('mesh.ply')
    body_list = mesh.split()
    edge_size = [item.vertices.shape[0] for item in body_list ]
    refined_mesh = body_list[edge_size.index(max(edge_size))]
    refined_mesh.export('refined_mesh.ply')
    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)