import os
import cv2
import numpy as np

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
