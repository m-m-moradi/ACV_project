import argparse
import os
from reconstruct import Multiview3DReconstructor

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

    reconstructor = Multiview3DReconstructor(
        img_dir=dataset,
        features_dir=features,
        downscale_factor=1,
    )
    reconstructor()
