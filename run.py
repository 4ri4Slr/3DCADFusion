import utils
import time
import numpy as np
import fusion
import trimesh
import predict_mask

if __name__ == "__main__":


    path = './data/gripper'
    res = [1500, 1800]
    d_t = 0.93

    utils.crop_images(path, res)
    predict_mask.predict(path, res)

    print("Estimating voxel volume bounds...")
    n_imgs = 100
    cam_intr = np.array([[6549.539, 0., 19],
                             [0., 6548.102, 772],
                             [0., 0., 1.]])
    vol_bnds = np.zeros((3, 2))

    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    ref_pcl = utils.load_pcl(path, 0)
    ref_depth_im = utils.clean_depth(ref_pcl['pointcloud'][:, :, 2], ref_pcl['mask'], res, d_t)
    cam_pose = np.loadtxt(path + "/frame-%05d.pose.txt" % 0)  # 4x4 rigid transformation matrix

    view_frust_pts = fusion.get_view_frustum(ref_pcl['pointcloud'][:, :, 2], cam_intr, cam_pose)
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.002)
    tsdf_vol.integrate(ref_pcl['color_image'], ref_depth_im, cam_intr,  cam_pose, obs_weight=1.)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs-1):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image and camera pose
        pcl1 =utils.load_pcl(path,i)
        pcl2 =utils.load_pcl(path,i+1)
        depth_im = utils.clean_depth(pcl2['pointcloud'][:, :, 2], pcl2['mask'], res, d_t)

        # Integrate observation into voxel volume (assume color aligned with depth)
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        cam_pose = utils.register_tsdf(verts, colors, pcl1, pcl2 , cam_pose , res,d_t)
        cam_pose = np.linalg.inv(cam_pose)
        tsdf_vol.integrate(pcl2['color_image'], depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    mesh = trimesh.load('mesh.ply')
    body_list = mesh.split()
    edge_size = [item.vertices.shape[0] for item in body_list]
    refined_mesh = body_list[edge_size.index(max(edge_size))]
    refined_mesh.export('refined_mesh.ply')
    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)
