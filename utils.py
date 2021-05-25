import read_binary_matrix
import numpy as np
from scipy.ndimage import zoom
import open3d as o3d
import os
import copy
import cv2
from numba import njit, prange
import matplotlib.pyplot as plt


def read_ply(path):
    pcd_load = o3d.io.read_point_cloud(path)
    return pcd_load


def downsample(path):

    pcl = read_binary_matrix.read_binary_matrix(path)
    pcl = zoom(pcl, (0.1, 0.1, 1))
    height, width = pcl.shape[:2]
    pcl = pcl.reshape(height*width, -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    o3d.io.write_point_cloud(os.path.splitext(path)[0] + ".ply", pcd)

def crop_center(img,cropx,cropy):
    x,y = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    if len(img.shape) == 3:
        return img[startx:startx+cropx,starty:starty+cropy,:]
    else:
        return img[startx:startx+cropx,starty:starty+cropy]
def npy2pcl(array):

    height, width = array.shape[:2]
    array = array.reshape(height * width, -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd

def visualize(pcl):
    #xyz = - np.asarray(pcl.points)
    #pcl.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcl])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      width = 1920,
                                      height = 1080,
                                      left = 50,
                                      top=50)


def register(source, target, threshold, trans_init):



    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    #reg_p2p = o3d.pipelines.registration.registration_colored_icp(
    #    source, target, threshold, trans_init,
    #    o3d.pipelines.registration.TransformationEstimationForColoredICP())

    print(reg_p2p.transformation)
    #draw_registration_result(source, target, reg_p2p.transformation)

    return reg_p2p.transformation


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh


def execute_fast_global_registration(source,target, voxel_size):

    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def remove_background_updated(array, mask, color,  ref, th_d =1.3, th_u = 2, ds_ratio = 0.001):

    height, width = array.shape[:2]
    pcl = array.copy()
    pcl[mask[:, :, 0] == 1, 2] = 0
    pcl[pcl[:,:,2] > th_u, 2] = 0
    pcl[pcl[:,:,2] < th_d, 2] = 0
    pcl[ (pcl[:,:,2] - ref[:,:,2])**2 + (pcl[:,:,1] - ref[:,:,1])**2 + (pcl[:,:,0] - ref[:,:,0])**2  < ds_ratio,2]= 0
    pcl = pcl.reshape(height*width, -1)
    #color = zoom(color, (ds_ratio, ds_ratio, 1))
    color = color.reshape(height*width,-1)/255
    remcells = np.delete(pcl, pcl[:, 2] > th_u, 0)
    color = np.delete(color, pcl[:, 2] > th_u,0)
    color = np.delete(color, remcells[:, 2] < th_d, 0)
    remcells = np.delete(remcells, remcells[:, 2] < th_d, 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(remcells)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals()
    #pcd_ret = [pcd, pcl, color]
    #pcd = pcd.voxel_down_sample(0.001)
    return pcd


def remove_background(array, mask, color, d_t, d_min = 0.5, ds_ratio = 1):

    array = zoom(array, (ds_ratio,ds_ratio,1))
    height, width = array.shape[:2]
    pcl = array.copy()
    #mask = zoom(mask, (ds_ratio,ds_ratio,1))
    pcl[mask[:, :, 0] == 1, 2] = 0
    pcl[pcl[:,:,2] > d_t, 2] = 0
    pcl[pcl[:,:,2] < d_min, 2] = 0
    pcl = pcl.reshape(height*width, -1)
    #color = zoom(color, (ds_ratio, ds_ratio, 1))
    color = color.reshape(height*width,-1)/255
    remcells = np.delete(pcl, pcl[:, 2] > d_t, 0)
    color = np.delete(color, pcl[:, 2] > d_t,0)
    color = np.delete(color, remcells[:, 2] < d_min, 0)
    remcells = np.delete(remcells, remcells[:, 2] < d_min, 0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(remcells)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals()
    #visualize(pcd)
    #pcd_ret = [pcd, pcl, color]
    pcd = pcd.voxel_down_sample(0.001)
    return pcd



def colored_icp(source,target, init, voxel_size):
    voxel_radius = [4*voxel_size,4*voxel_size, 2*voxel_size, voxel_size]
    max_iter = [100, 50, 30, 14]
    current_transformation = init
    for scale in range(4):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.3f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        draw_registration_result(source_down, target_down, current_transformation)
    return current_transformation




def execute_registeration_sequence(path, d_t, res = [1500,1800]):

    start = 0
    pcl_ref = np.load(path + '/xyz/' + str(start) + '.npy')
    mask_ref = cv2.imread(path + '/masks/frame-' + str(start).zfill(6) + '.color.jpg')
    rgb_ref = cv2.imread(path + '/rectified_rgb/image' + str(start+1).zfill(5) + '.png')

    pcd_ref = crop_center(pcl_ref,  res[0], res[1])
    pcd_ref = remove_background(pcd_ref, mask_ref, rgb_ref, d_t)
    pcd_ref1 = pcd_ref
    t_prev = np.eye(4)
    t_ref = t_prev
    pcd_combined = pcd_ref1
    pcd_combined_pts = np.asarray(pcd_combined.points)
    pcd_combined_scores = np.zeros(len(pcd_combined_pts))


    for i in range(start,90):
        print(i)

        pcl1 = np.load(path + '/xyz/' + str(i) + '.npy')
        pcl2 = np.load(path + '/xyz/' + str(i+1) + '.npy')
        mask1 = cv2.imread(path + '/masks/frame-' + str(i).zfill(6) + '.color.jpg')
        mask2 = cv2.imread(path + '/masks/frame-' + str(i+1).zfill(6) + '.color.jpg')
        rgb1 = cv2.imread(path + '/rectified_rgb/image' + str(i+1).zfill(5) + '.png')
        rgb2 = cv2.imread(path + '/rectified_rgb/image' + str(i + 2).zfill(5) + '.png')
        #rgb2[:, :, 2] = 0
        pcd1 = crop_center(pcl1, res[0], res[1])
        pcd2 = crop_center(pcl2,  res[0], res[1])

        pcd1 = remove_background(pcd1, mask1, rgb1, d_t)
        pts = np.asarray(pcd1.points)
        print(pts.shape)
        pcd2 = remove_background(pcd2, mask2, rgb2, d_t)


        pcd1, ind = pcd1.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=1.3)
        pcd2, ind = pcd2.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=1.3)
        #visualize(pcd1)

        evaluation = o3d.pipelines.registration.evaluate_registration(pcd1, pcd2, 0.005, np.eye(4))
        #print(evaluation)
        corre_ratio = np.asarray(evaluation.correspondence_set).shape[0] / np.asarray(pcd1.points).shape[0]
        print(corre_ratio)
        if corre_ratio < 0.7:
            result = execute_fast_global_registration(pcd1,pcd2,0.0005)
            draw_registration_result(pcd1,pcd2, result.transformation)
            t_init = result.transformation
        else:
            t_init = np.eye(4)
        print(t_init)
        #t_mat = colored_icp(pcd1, pcd2, t_init, 0.0008)
        #draw_registration_result(pcd1, pcd2, np.eye(4))
        t_mat = register(pcd1, pcd2, 0.01,t_init)
        #draw_registration_result(pcd1, pcd2, t_mat)
        #t_mat = register(pcd1, pcd2, 0.001, t_mat)

        #if i >15: draw_registration_result(pcd1, pcd2, t_mat)
        t_mat = register(pcd1, pcd2, 0.005, t_mat)
        #if i > 15: draw_registration_result(pcd1, pcd2, t_mat)
        t_ref = t_mat @ t_ref
        t_ref = register(pcd_combined, pcd2, 0.01, t_ref)
        t_ref = register(pcd_combined, pcd2, 0.005, t_ref)
        #t_ref = colored_icp(pcd_combined, pcd2, t_ref, 0.0008)

        #t_ref = colored_icp(pcd_combined, pcd2, t_ref, 0.005)
        #draw_registration_result(pcd_combined, pcd2, t_ref)
        #corres = np.asarray(evaluation.correspondence_set)
        #pcd_combined_pts = np.asarray(pcd_combined.points)
        #pcd_combined_scores[corres[:, 0]] += 1
        #pcd2_pts = np.asarray(pcd2.points)
        #newss = np.delete(pcd2_pts,pcd2_pts[corres[:,1]])
        #pcd_combined_pts = np.append(pcd_combined_pts, np.delete(pcd2_pts,corres[:,1] ))

        #pcd_combined_scores = np.append(pcd_combined_pts, np.zeros(len(np.delete(pcd2_pts,corres[:,1] ))))
        #if i >20: draw_registration_result(pcd_combined,pcd2, np.eye(4))
        if i % 5 == 0 :
            #pcd_combined_pts = np.delete(pcd_combined_pts[pcd_combined_scores < 2])
            #pcd_combined_scores = np.delete(pcd_combined_scores[pcd_combined_scores < 2])

            #pcd_combined = o3d.geometry.PointCloud()
            #pcd_combined.points = o3d.utility.Vector3dVector(pcd_combined_pts)
            #visualize(pcd_combined)
            if i == 5:
                continue
            pcd2.transform(np.linalg.inv(t_ref))
            pcd2, ind = pcd2.remove_statistical_outlier(nb_neighbors=100,
                                                   std_ratio=1.3)
            pcd_temp = copy.deepcopy(pcd2)
            pcd_combined += pcd_temp
            pcd_combined = pcd_combined.voxel_down_sample(0.001)
            pcd_combined, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=50,
                                                   std_ratio=4)
            #visualize(pcd_combined)
        with open(path + 'frame-' + str(i+1).zfill(5) + '.pose.txt', "w") as f:
            np.savetxt(f, t_ref)

    o3d.io.write_point_cloud('%d_combined.ply' % i, pcd_combined)


# def execute_registeration_sequence_updated(dpath = './xyz/', mpath = './masks/', rgbpath = './rectified_rgb/', loop = 5):
#
#     pcl_ref = np.load(dpath + str(0) + '.npy')
#     mask_ref = cv2.imread(mpath + 'frame-' + str(0).zfill(6) + '.color.jpg')
#     rgb_ref = cv2.imread(rgbpath + 'image' + str(1).zfill(5) + '.png')
#
#     pcd_ref = crop_center(pcl_ref, 1500, 1800)
#     pcd_ref, pcl_ref = remove_background(pcd_ref, mask_ref, rgb_ref)
#     pcd_ref1 = pcd_ref
#     t_prev = np.eye(4)
#     t_ref = t_prev
#     pcd_combined = pcd_ref1
#     pcd_combined_pts = np.asarray(pcd_combined.points)
#     pcd_combined_scores = np.zeros(len(pcd_combined_pts))
#     for i in range(95):
#         print(i)
#         pcl1 = np.load(dpath + str(i) + '.npy')
#         pcl2 = np.load(dpath + str(i+1) + '.npy')
#         mask1 = cv2.imread(mpath + 'frame-' + str(i).zfill(6) + '.color.jpg')
#         mask2 = cv2.imread(mpath + 'frame-' + str(i+1).zfill(6) + '.color.jpg')
#         rgb1 = cv2.imread(rgbpath + 'image' + str(i+1).zfill(5) + '.png')
#         rgb2 = cv2.imread(rgbpath + 'image' + str(i + 2).zfill(5) + '.png')
#         #rgb2[:, :, 2] = 0
#         pcd1 = crop_center(pcl1, 1500, 1800)
#         pcd2 = crop_center(pcl2, 1500, 1800)
#
#         pcd1, pcl1 = remove_background(pcd1, mask1, rgb1)
#         pcd2, pcl2 = remove_background(pcd2, mask2, rgb2)
#         t_mat = register(pcd1, pcd2, 0.05, np.eye(4))
#         t_mat = register(pcd1, pcd2, 0.01, t_mat)
#         t_mat = register(pcd1, pcd2, 0.005, t_mat)
#
#         evaluation = o3d.pipelines.registration.evaluate_registration(
#             pcd1, pcd2, 0.01, t_mat)
#         print(evaluation)
#
#         t_ref = t_mat @ t_ref
#         t_ref = register(pcd_combined, pcd2, 0.01, t_ref)
#         t_ref = register(pcd_combined, pcd2, 0.005, t_ref)
#
#
#         # Merging New Points: Add new points
#         # Update Weights
#         # Remove Points
#         #
#
#         pcd_combined_pts = np.asarray(pcd_combined.points)
#
#         pix = cam2pix(np.asarray(pcd_combined_pts)
#         #pix_x =  pcd_combined_pts[pix]
#         #if i > 24: draw_registration_result(pcd_combined, pcd2, t_ref)
#
#         #pcd2.transform(np.linalg.inv(t_ref))
#         #evaluation = o3d.pipelines.registration.evaluate_registration(
#         #    pcd_combined, pcd2, 0.005, t_ref)
#         #print(evaluation)
#         #corres = np.asarray(evaluation.correspondence_set)
#         #pcd_combined_pts = np.asarray(pcd_combined.points)
#         #pcd_combined_scores[corres[:, 0]] += 1
#         #pcd2_pts = np.asarray(pcd2.points)
#         #pcd2_colors = np.asarray(pcd2.colors)
#         #visualize(pcd2)
#         #newss = np.delete(pcd2_pts,pcd2_pts[corres[:,1]])
#         #pcd_combined_pts = np.append(pcd_combined_pts, np.delete(pcd2_pts,corres[:,1], axis = 0 ))
#
#         #pcd_combined_scores = np.append(pcd_combined_pts, np.zeros(len(np.delete(pcd2_pts,corres[:,1] ))))
#
#         if i % 5 == 0:
#             #pcd_combined_pts = np.delete(pcd_combined_pts[pcd_combined_scores < 2])
#             #pcd_combined_scores = np.delete(pcd_combined_scores[pcd_combined_scores < 2])
#
#         #pcd_combined = o3d.geometry.PointCloud()
#         #pcd_combined.points = o3d.utility.Vector3dVector(pcd_combined_pts)
#             #visualize(pcd_combined)
#             #pcd_temp = o3d.geometry.PointCloud()
#             #pcd_temp.points = o3d.utility.Vector3dVector(pcd2_pts[corres[:,1]])
#             #pcd_temp.colors = o3d.utility.Vector3dVector(pcd2_colors[corres[:,1]])
#             #pcd_temp.estimate_normals()
#             #visualize(pcd_temp)
#             #print('updating')
#             #pcd_temp, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=100,
#             #                                          std_ratio=1.5)
#             #pcd_combined += pcd_temp
#             #pcd_combined = pcd_combined.voxel_down_sample(0.001)
#             #pcd_combined, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=50,
#             #                                       std_ratio=4)
#         #draw_registration_result(pcd_combined, pcd2, np.eye(4))
#         if i%15 == 0:
#             #draw_registration_result(pcd_combined, pcd2, np.eye(4))
#             visualize(pcd_combined)
#             #if i%5 == 0:
#             #    draw_registration_result(pcd_ref1, pcd2, t_ref)
#
#         with open('./data/frame-' + str(i+1).zfill(5) + '.pose.txt', "w") as f:
#             np.savetxt(f, t_ref)
#
#     o3d.io.write_point_cloud('%d_combined.ply' % i, pcd_combined)
#
#
# def cam2pix(cam_pts,intr =np.loadtxt( "data/camera-intrinsics.txt", delimiter=' '), ds_ratio=5):
#     """Convert camera coordinates to pixel coordinates.
#     """
#     intr = intr.astype(np.float32)
#     fx, fy = intr[0, 0]/ds_ratio, intr[1, 1]/ds_ratio
#     cx, cy = intr[0, 2]/ds_ratio, intr[1, 2]/ds_ratio
#     pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
#     for i in prange(cam_pts.shape[0]):
#       pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
#       pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
#     return pix


def execute_depth_generation_sequence(path= './xyz/', d = 380, res = [1600,1200]):

    for i in range(len(os.listdir(path + 'xyz/'))):
        print(len(os.listdir(path + 'xyz/')))
        pcl = np.load(path + 'xyz/' + str(i) + '.npy')
        pcd = crop_center(pcl, res[0], res[1])
        mask = cv2.imread( path + 'masks/frame-' + str(i).zfill(6) + '.color.jpg')
        #pcd = zoom(pcd, (1200, 1500, 1))
        depth = pcd[:, :, 2]
        depth = depth*255
        depth[mask[:, :, 0] == 1] = 0
        depth[depth > d] = 0
        cv2.imwrite(path + 'frame-' + str(i+1).zfill(5) + '.depth.png', depth)

def rename(path, height, width):

    for i in range(1,51):
        img = read_binary_matrix.read_binary_matrix(path + str(i) + '.bin')
        img = crop_center(img, height, width)
        #img[:,:,:3] = img
        img2 = np.zeros([img.shape[0], img.shape[1],3])
        for j in range(3):
            img2[:,:, j]  = img[:,:,0]
        img2 = img2*255
        cv2.imwrite('./rectified_rgb/image' + str(i ).zfill(5) + '.png', img2)
        #plt.imsave('./depth_render/frame-' + str(i - 1).zfill(6) + '.color.jpg', img2)


def clean_depth(depth_im, mask, res, d = 1.5):

    depth_im = crop_center(depth_im, res[0], res[1])
    depth_im[mask[:, :, 0] == 1] = 0
    depth_im[depth_im > d] = 0
    #cv2.imshow('name',depth_im)
    #depth_im = cv2.bilateralFilter(depth_im,9,30,30)
    #cv2.imshow('', depth_im)

    return depth_im

def register_tsdf(verts, colors, depth_im, color_image, mask, cam_intr, cam_pose, res = [1600,1200]):


    target = crop_center(depth_im, res[0], res[1])
    target = remove_background(target, mask, color_image)

    pcl_global = o3d.geometry.PointCloud()
    pcl_global.points = o3d.utility.Vector3dVector(verts)
    #target.colors = o3d.utility.Vector3dVector(colors)
    pcl_global.estimate_normals()
    #pcd_down = pcl_global.voxel_down_sample(0.003)
    target_down = target.voxel_down_sample(0.002)
    pose = register(pcl_global, target_down, 0.05, np.linalg.inv(cam_pose))
    #draw_registration_result(pcl_global, target_down, pose)
    pose = register(pcl_global,target,  0.003, pose)
    #pose = register(pcl_global, target, 0.0001, pose)
    #visualize(pcl_global)
    draw_registration_result(pcl_global, target, pose)

    return pose
    # Flip it, otherwise the pointcloud will be upside down

