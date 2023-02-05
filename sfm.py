import cv2
import numpy as np
import os
import argparse
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt


class Image_loader:
    def __init__(self, img_dir: str, downscale_factor: float):
        # loading the Camera intrinsic parameters K
        K_path = os.path.join(img_dir, "K.txt")
        self.K = np.loadtxt(K_path)

        # Loading the set of images
        self.image_list = []
        for image_file_name in sorted(os.listdir(img_dir)):
            if (
                image_file_name[-4:].lower() == ".jpg"
                or image_file_name[-5:].lower() == ".png"
            ):
                image_file_path = os.path.join(img_dir, image_file_name)
                self.image_list.append(image_file_path)

        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale()

    def downscale(self) -> None:
        """
        Downscales the Image intrinsic parameter acc to the downscale factor
        """
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor

    def downscale_image(self, image):
        for _ in range(1, int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image


class Sfm:
    def __init__(self, img_dir: str, features_dir: str = "", downscale_factor: float = 2.0) -> None:
        """
        Initialise and Sfm object.
        """
        self.image_loader = Image_loader(img_dir, downscale_factor)
        self.features_dir = features_dir
        self.use_deep = False
        if features_dir:
            self.use_deep = True

    def triangulation(self, projection_matrix_1, projection_matrix_2, point_2d_1, point_2d_2) -> tuple:
        """
        Triangulates 3d points from 2d vectors and projection matrices
        returns point cloud
        see: https://docs.opencv.org/4.5.1/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
        """
        pt_cloud = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, point_2d_1.T, point_2d_2.T)
        # pt_cloud: 4xN array of reconstructed points in homogeneous coordinates. These points are returned in the world's coordinate system.
        return (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point, K, dist_coeff) -> tuple:
        """
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        """

        # see this: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        # see this: https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html

        _, rot_vector_calc, tran_vector, inliers = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)

        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)
        return rot_matrix, tran_vector, inliers


    def reprojection_error(self, obj_points, image_points, projection_matrix, K) -> tuple:
        """
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        """

        rot_matrix = projection_matrix[:3, :3]
        tran_vector = projection_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])                              # change shape to (N, 2)
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T), cv2.NORM_L2)      # images_points is (2, N)

        return total_error / image_points_calc.shape[0]


    def to_ply(self, path, point_cloud, colors) -> None:
        """
        Generates the .ply which can be used to open the point cloud
        """
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
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3
        """
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        # print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2


    def find_features_deep(self, image_0_path, image_1_path) -> tuple:
        # print(image_0_path, image_1_path)
        # breakpoint()
        image_0_name = os.path.splitext(os.path.basename(image_0_path))[0]
        image_1_name = os.path.splitext(os.path.basename(image_1_path))[0]
        path = os.path.join(self.features_dir, f"{image_0_name}_{image_1_name}_matches.npz")
        npz = np.load(path)
        keypoints0 = npz['keypoints0']
        keypoints1 = npz['keypoints1']
        matches = npz['matches']
        features0 = np.array([keypoints0[i] for i in range(keypoints0.shape[0]) if matches[i] > -1], dtype='float32')
        features1 = np.array([keypoints1[matches[i]] for i in range(matches.shape[0]) if matches[i] > -1], dtype='float32')
        assert len(features0) == len(features1)
        return features0, features1


    def find_features_SIFT(self, image_0, image_1) -> tuple:
        """
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        """

        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        # Brute Forse Matching
        # see: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        features0 = np.float32([key_points_0[m.queryIdx].pt for m in feature])
        features1 = np.float32([key_points_1[m.trainIdx].pt for m in feature])
        return features0, features1

    def __call__(self):
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        pose_array = self.image_loader.K.ravel()

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
            feature_0, feature_1 = self.find_features_deep(image_0_path, image_1_path)
        else:
            feature_0, feature_1 = self.find_features_SIFT(image_0, image_1)

        # Essential matrix
        # We recover E because we have K from our datasets
        # breakpoint()
        essential_matrix, em_mask = cv2.findEssentialMat(
            feature_0,
            feature_1,
            self.image_loader.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.4,
            mask=None,
        )

        # filter out outlier features
        # using features that contribute in calculating essential matrix
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        # see: https://stackoverflow.com/questions/74029853/opencv-what-does-it-mean-when-the-number-of-inliers-returned-by-recoverpose-f
        # Decompose E to R,t
        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.image_loader.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]

        PI_matrix_1[:3, :3] = rot_matrix
        PI_matrix_1[:3, 3] = tran_matrix.ravel()
        projection_matrix_1 = np.matmul(self.image_loader.K, PI_matrix_1)

        # breakpoint()
        points_3d = self.triangulation(projection_matrix_0, projection_matrix_1, feature_0, feature_1)  # shape: (4, N)
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)                                       # shape: (N, 1, 3)
        error = self.reprojection_error(points_3d, feature_1.T, PI_matrix_1, self.image_loader.K)
        points_3d = points_3d[:, 0, :]                                                                  # shape: (N, 3)

        # ideally error < 1
        # print("REPROJECTION ERROR: ", error)

        _, _, inliers = self.PnP(
            obj_point=points_3d,
            image_point=feature_1,
            K=self.image_loader.K,
            dist_coeff=np.zeros((5, 1), dtype=np.float32),
        )
        if inliers is not None:
            inliers = inliers.ravel()
            points_3d = points_3d[inliers]
            feature_0 = feature_0[inliers]
            feature_1 = feature_1[inliers]


        total_images = len(self.image_loader.image_list) - 2
        pose_array = np.hstack((np.hstack((pose_array, projection_matrix_0.ravel())), projection_matrix_1.ravel()))

        for i in tqdm(range(total_images)):
            image_2_path = self.image_loader.image_list[i + 2]
            image_2 = self.image_loader.downscale_image(cv2.imread(image_2_path))

            if self.use_deep:
                features_cur, features_2 = self.find_features_deep(image_1_path, image_2_path)
            else:
                features_cur, features_2 = self.find_features_SIFT(image_1, image_2)

            # features_cur, features_2 = self.find_features_SIFT(image_1, image_2)

            if i != 0:
                points_3d = self.triangulation(projection_matrix_0, projection_matrix_1, feature_0, feature_1)
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]
            points_3d = points_3d[cm_points_0]

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


            PI_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            projection_matrix_2 = np.matmul(self.image_loader.K, PI_matrix_1)


            points_3d = self.triangulation(projection_matrix_1, projection_matrix_2, cm_mask_0, cm_mask_1)
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)

            cm_mask_0 = cm_mask_0.T
            cm_mask_1 = cm_mask_1.T
            error = self.reprojection_error(
                points_3d,
                cm_mask_1,
                PI_matrix_1,
                self.image_loader.K,
            )


            # print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, projection_matrix_2.ravel()))

            total_points = np.vstack((total_points, points_3d[:, 0, :]))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
            total_colors = np.vstack((total_colors, color_vector))

            PI_matrix_0 = np.copy(PI_matrix_1)
            projection_matrix_0 = np.copy(projection_matrix_1)

            # plt.scatter(i, error)
            # plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            image_1_path = image_2_path
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            projection_matrix_1 = np.copy(projection_matrix_2)


        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.image_loader.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(
            self.image_loader.path
            + "/res/"
            + self.image_loader.image_list[0].split("/")[-2]
            + "_pose_array.csv",
            pose_array,
            delimiter="\n",
        )


def valid_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("The file path does not exist")
    return path


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=valid_path, required=True, help='path of dataset (directory) to create the 3d reconstruction')
parser.add_argument('-f', '--features', type=valid_path, default=None, help='path of features (directory) that features are pre-created via a deep model')

if __name__ == "__main__":
    args = parser.parse_args()
    features = args.features
    dataset = args.dataset

    sfm = Sfm(
        img_dir=dataset,
        features_dir=features,
        downscale_factor=1,
    )
    sfm()
