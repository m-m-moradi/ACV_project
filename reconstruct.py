import cv2
import os
import numpy as np
from tqdm import tqdm
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import *

from utils import Image_loader

def load_torch_image(fname):
    img = cv2.imread(fname)

    factor = 3
    height, width = img.shape[:2]
    new_height, new_width = int(height / factor), int(width / factor)
    img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img


class Multiview3DReconstructor:

    def __init__(self, img_dir: str, features_dir: str = "", downscale_factor: float = 2.0) -> None:
        self.image_loader = Image_loader(img_dir, downscale_factor)
        self.features_dir = features_dir
        self.use_deep = False
        if features_dir:
            self.use_deep = True

    def triangulation(self, projection_matrix_1, projection_matrix_2, point_2d_1, point_2d_2) -> tuple:
        # see: https://docs.opencv.org/4.5.1/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
        # pt_cloud: 4xN array of reconstructed points in homogeneous coordinates. These points are returned in the world's coordinate system.

        pt_cloud = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, point_2d_1.T, point_2d_2.T)
        return (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point, K, dist_coeff) -> tuple:
        # see this: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        # see this: https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html

        _, rot_vector_calc, tran_vector, inliers = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)
        return rot_matrix, tran_vector, inliers


    def reprojection_error(self, obj_points, image_points, projection_matrix, K) -> tuple:

        rot_matrix = projection_matrix[:3, :3]
        tran_vector = projection_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])                              # change shape to (N, 2)
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T), cv2.NORM_L2)      # images_points is (2, N)

        return total_error / image_points_calc.shape[0]


    def to_ply(self, path, point_cloud, colors) -> None:

        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(
            scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2
        )
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = """ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            """
        with open(
            path + "/res/" + self.image_loader.image_list[0].split("/")[-2] + ".ply",
            "w",
        ) as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, "%f %f %f %d %d %d")


    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        """
        Find the common points in three arrays of 3D image points and return the indices 
        of the common points and the filtered arrays.
        """

        index1_cm12 = []
        index2_cm12 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                index1_cm12.append(i)
                index2_cm12.append(a[0][0])

        image_points_1_cm12 = np.ma.array(image_points_2, mask=True)
        image_points_1_cm12.mask[index2_cm12] = False

        image_points2_cm23 = np.ma.array(image_points_3, mask=True)
        image_points2_cm23.mask[index2_cm12] = False

        return np.array(index1_cm12), np.array(index2_cm12), image_points_1_cm12.data, image_points2_cm23.data


    def find_correspondence_deep(self, image_0_path, image_1_path) -> tuple:
        # print(image_0_path, image_1_path)
        # image_0_name = os.path.splitext(os.path.basename(image_0_path))[0]
        # image_1_name = os.path.splitext(os.path.basename(image_1_path))[0]
        # path = os.path.join(self.features_dir, f"{image_0_name}_{image_1_name}_matches.npz")
        # npz = np.load(path)
        # keypoints0 = npz['keypoints0']
        # keypoints1 = npz['keypoints1']
        # matches = npz['matches']
        # match_confidence = npz['match_confidence']
        # features0 = np.array([keypoints0[i] for i in range(keypoints0.shape[0]) if match_confidence[i] >= 0.95], dtype='float32')
        # features1 = np.array([keypoints1[matches[i]] for i in range(matches.shape[0]) if match_confidence[i] >= 0.95], dtype='float32')
        # assert len(features0) == len(features1)
        # # breakpoint()
        # return features0, features1
        img1 = load_torch_image(image_0_path)
        img2 = load_torch_image(image_1_path)


        matcher = KF.LoFTR(pretrained='outdoor')
        input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.inference_mode():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 10000)
        inliers = inliers > 0
        return mkpts0[inliers.ravel()], mkpts1[inliers.ravel()]


    def find_correspondence_SIFT(self, image_0, image_1) -> tuple:
        """
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        """

        sift = cv2.SIFT_create(edgeThreshold=5000)
        # sift = cv2.xfeatures2d.SURF_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        # Brute Forse Matching
        # see: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.90 * n.distance:
                feature.append(m)

        corr1 = np.array([key_points_0[m.queryIdx].pt for m in feature], dtype='float32')
        corr2 = np.array([key_points_1[m.trainIdx].pt for m in feature], dtype='float32')
        _, em_mask = cv2.findEssentialMat(
            corr1,
            corr2,
            self.image_loader.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.4,
            mask=None,
        )

        # TODO: use E to Dense matching, this is not dense matching it is saying which corrs are really corss
        # filter out outlier features
        # using features that contribute in calculating essential matrix
        corr1 = corr1[em_mask.ravel() == 1]
        corr2 = corr2[em_mask.ravel() == 1]

        assert len(corr1) == len(corr2)
        return corr1, corr2


    def __call__(self):

        PI_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        PI_matrix_1 = np.zeros((3, 4))

        projection_matrix_0 = np.matmul(self.image_loader.K, PI_matrix_0)
        projection_matrix_1 = np.empty((3, 4))

        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0_path = self.image_loader.image_list[0]
        image_1_path = self.image_loader.image_list[1]
        image_0 = self.image_loader.downscale_image(cv2.imread(image_0_path))
        image_1 = self.image_loader.downscale_image(cv2.imread(image_1_path))

        if self.use_deep:
            corr_0, corr_1 = self.find_correspondence_deep(image_0_path, image_1_path)
        else:
            corr_0, corr_1 = self.find_correspondence_SIFT(image_0, image_1)

        # Essential matrix
        # We recover E because we have K from our datasets
        # breakpoint()
        essential_matrix, em_mask = cv2.findEssentialMat(
            corr_0,
            corr_1,
            self.image_loader.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.4,
            mask=None,
        )

        # filter out outlier corrs
        # using corrs that contribute in calculating essential matrix
        corr_0 = corr_0[em_mask.ravel() == 1]
        corr_1 = corr_1[em_mask.ravel() == 1]

        # see: https://stackoverflow.com/questions/74029853/opencv-what-does-it-mean-when-the-number-of-inliers-returned-by-recoverpose-f
        # Decompose E to R,t
        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, corr_0, corr_1, self.image_loader.K)
        corr_0 = corr_0[em_mask.ravel() > 0]
        corr_1 = corr_1[em_mask.ravel() > 0]

        PI_matrix_1[:3, :3] = rot_matrix
        PI_matrix_1[:3, 3] = tran_matrix.ravel()
        projection_matrix_1 = np.matmul(self.image_loader.K, PI_matrix_1)

        # breakpoint()
        points_3d = self.triangulation(projection_matrix_0, projection_matrix_1, corr_0, corr_1)  # shape: (4, N)
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)                                       # shape: (N, 1, 3)
        # error = self.reprojection_error(points_3d, corr_1.T, PI_matrix_1, self.image_loader.K)
        points_3d = points_3d[:, 0, :]                                                                  # shape: (N, 3)

        # ideally error < 1
        # print("REPROJECTION ERROR: ", error)

        _, _, inliers = self.PnP(
            obj_point=points_3d,
            image_point=corr_1,
            K=self.image_loader.K,
            dist_coeff=np.zeros((5, 1), dtype=np.float32),
        )
        if inliers is not None:
            inliers = inliers.ravel()
            points_3d = points_3d[inliers]
            corr_0 = corr_0[inliers]
            corr_1 = corr_1[inliers]


        total_images = len(self.image_loader.image_list) - 2
        for i in tqdm(range(total_images)):
            image_2_path = self.image_loader.image_list[i + 2]
            image_2 = self.image_loader.downscale_image(cv2.imread(image_2_path))

            if self.use_deep:
                corrs_cur, corrs_2 = self.find_correspondence_deep(image_1_path, image_2_path)
            else:
                corrs_cur, corrs_2 = self.find_correspondence_SIFT(image_1, image_2)

            if i != 0:
                points_3d = self.triangulation(projection_matrix_0, projection_matrix_1, corr_0, corr_1)
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]

            index1_cm1curr, indexcurr_cm1curr, corr1_cm1curr, corr2_cmcurr3 = self.common_points(corr_1, corrs_cur, corrs_2)
            points_3d = points_3d[index1_cm1curr]           # give me 3D points that are connected to corr_curr that I have (why index1? becasue 3D points and corr1 match)
            cm_points_cur = corrs_cur[indexcurr_cm1curr]    # give me corrs to above
            cm_points_2 = corrs_2[indexcurr_cm1curr]

            rot_matrix, tran_matrix, inliers = self.PnP(
                obj_point=points_3d,
                image_point=cm_points_2,
                K=self.image_loader.K,
                dist_coeff=np.zeros((5, 1), dtype=np.float32),
            )
            if inliers is not None:
                inliers = inliers.ravel()
                points_3d = points_3d[inliers]
                cm_points_2 = cm_points_2[inliers]
                cm_points_cur = cm_points_cur[inliers]

            PI_matrix_2 = np.hstack((rot_matrix, tran_matrix))
            projection_matrix_2 = np.matmul(self.image_loader.K, PI_matrix_2)


            points_3d = self.triangulation(projection_matrix_1, projection_matrix_2, corr1_cm1curr, corr2_cmcurr3)
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)

            corr1_cm1curr = corr1_cm1curr.T
            corr2_cmcurr3 = corr2_cmcurr3.T
            error = self.reprojection_error(
                points_3d,
                corr2_cmcurr3,
                PI_matrix_1,
                self.image_loader.K,
            )
            # print("Reprojection Error: ", error)
            total_points = np.vstack((total_points, points_3d[:, 0, :]))
            points_left = np.array(corr2_cmcurr3, dtype=np.int32)
            color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
            total_colors = np.vstack((total_colors, color_vector))
            projection_matrix_0 = np.copy(projection_matrix_1)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            image_1_path = image_2_path
            corr_0 = np.copy(corrs_cur)
            corr_1 = np.copy(corrs_2)
            projection_matrix_1 = np.copy(projection_matrix_2)


        print("Printing to .ply file")
        total_points = total_points - np.median(total_points, axis=0)
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.image_loader.path, total_points, total_colors)
        print("Completed Exiting ...")
