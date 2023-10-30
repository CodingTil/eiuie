import glob
import os
from typing import Dict, Tuple

import numpy as np
import cv2


def get_image_pair(image_name: str) -> Tuple[str, str]:
    """
    Get a pair of images, one original and one processed.

    Parameters
    ----------
    image_name : str
        Image name, without extension.

    Returns
    -------
    Tuple[str, str]
        Tuple of absolute paths to the images.
    """
    assert os.path.exists(
        f"data/fivek/raw/{image_name}.jpg"
    ), f"Image {image_name} does not exist."
    abs_path = os.path.abspath(f"data/fivek/raw/{image_name}.jpg")
    raw_image = cv2.imread(abs_path)
    dimensions = raw_image.shape[:2]

    files = glob.glob(f"data/fivek/*/{image_name}.jpg")
    files.remove(f"data/fivek/raw/{image_name}.jpg")
    assert len(files) >= 1, f"Image {image_name} does not have any processed images."
    for file in list(files):
        img = cv2.imread(file)
        if img.shape[:2] != dimensions:
            files.remove(file)
    assert (
        len(files) >= 1
    ), f"Image {image_name} does not have any processed images with the same dimensions."

    index = hash(image_name) % len(files)
    processed_image = files[index]

    return abs_path, os.path.abspath(processed_image)


def get_all_image_pairs() -> Dict[str, Tuple[str, str]]:
    """
    Get all image pairs.

    Returns
    -------
    Dict[str, Tuple[str, str]]
        Dictionary of image pairs, with image name as key and tuple of absolute paths as value.
    """
    image_names = [
        os.path.splitext(os.path.basename(file))[0]
        for file in glob.glob("data/fivek/raw/*.jpg")
    ]
    result = {}
    for image_name in image_names:
        try:
            result[image_name] = get_image_pair(image_name)
        except AssertionError as e:
            print(e)
    return result
