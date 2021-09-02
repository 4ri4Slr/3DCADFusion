import numpy as np
import open3d as o3d
import os
import copy
import cv2
import logging

def crop_center(img,cropx,cropy):

    x,y = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    if len(img.shape) == 3:
        return img[startx:startx+cropx,starty:starty+cropy,:]
    else:
        return img[startx:startx+cropx,starty:starty+cropy]


def visualize(pcl):
    o3d.visualization.draw_geometries([pcl])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      width = 1920,
                                      height = 1080,
                                      left = 50,
                                      top=50)




def preprocess_point_cloud(pcd, voxel_size):
    logging.info(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    logging.info(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    logging.info(":: Compute FPFH feature with search radius %.3f." % radius_feature)
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
    logging.info(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def remove_background(pcld, d_max, d_min = 0.1):

    array = pcld['pointcloud']
    mask = pcld['mask']
    color = pcld['color_image']
    height, width = array.shape[:2]
    pcl = array.copy()
    pcl[mask[:, :, 0] == 1, 2] = 0
    pcl[pcl[:, :, 2] > d_max, 2] = 0
    pcl[pcl[:, :, 2] < d_min, 2] = 0
    pcl = pcl.reshape(height*width, -1)
    color = color.reshape(height*width,-1)/255
    remcells = np.delete(pcl, pcl[:, 2] > d_max, 0)
    color = np.delete(color, pcl[:, 2] > d_max,0)
    color = np.delete(color, remcells[:, 2] < d_min, 0)
    remcells = np.delete(remcells, remcells[:, 2] < d_min, 0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(remcells)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals()
    pcd = pcd.voxel_down_sample(0.001)
    return pcd



def register_icp(source,target, init, voxel_size):
    voxel_radius = [6*voxel_size,4*voxel_size, 2*voxel_size, voxel_size]
    max_iter = [60, 40, 30, 14]
    current_transformation = init
    for scale in range(4):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        logging.info([iter, radius, scale])

        logging.info("3-1. Downsample with a voxel size %.3f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        logging.info("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 1, max_nn=30))

        logging.info("3-3. Applying icp point cloud registration")
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        #draw_registration_result(source_down, target_down, current_transformation)
    return current_transformation


def load_pcl(path,i):

    color_image = cv2.cvtColor(cv2.imread(path + "/rectified_rgb/%d.png" % (i)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(path + '/masks/frame-' + str(i).zfill(6) + '.color.jpg')
    xyz = np.load(path + '/xyz/%d.npy' % i)
    pcl = {"pointcloud":xyz, "color_image": color_image, "mask": mask}
    return pcl


def crop_images(path, res):

    ids = os.listdir(path)
    ids.sort(key=lambda x: int(os.path.splitext(x)[0]))
    images_fps = [os.path.join(path, image_id) for image_id in ids]
    for i, id in enumerate(ids):

            img = cv2.imread(images_fps[i])

            if img.shape[:2] == tuple(res):
                continue

            img = crop_center(img, res[0], res[1])
            cv2.imwrite(os.path.join(path, '%d.png' %int(os.path.splitext(id)[0])), img)



def clean_depth(depth_im, mask, res, d):

    depth_im = crop_center(depth_im, res[0], res[1])
    depth_im[mask[:, :, 0] == 1] = 0
    depth_im[depth_im > d] = 0

    return depth_im

def register_tsdf(verts,pcl1, pcl2, cam_pose, res, d_t):


    pcl1['pointcloud'] = crop_center(pcl1['pointcloud'], res[0], res[1])
    pcd1 = remove_background(pcl1, d_max = d_t)
    pcl2['pointcloud'] = crop_center(pcl2['pointcloud'], res[0], res[1])
    pcd2 = remove_background(pcl2, d_max = d_t)
    t_mat = register_icp(pcd1, pcd2, np.eye(4),0.002*1.5)
    cam_pose = t_mat @ np.linalg.inv(cam_pose)
    pcl_global = o3d.geometry.PointCloud()
    pcl_global.points = o3d.utility.Vector3dVector(verts)
    pcl_global.estimate_normals()
    pose = register_icp(pcl_global, pcd2, cam_pose, 0.002)

    return pose
