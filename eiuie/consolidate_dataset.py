from typing import Generator, Dict, Literal
import os

import numpy as np
import cv2
import glob

import pixel_dataset as pxds


def prepare_dataset() -> None:
    generator = __consolidate_data()
    if os.path.exists(pxds.FILE):
        os.remove(pxds.FILE)
    with open(pxds.FILE, "wb") as file:
        for data in generator:
            combined = np.hstack(
                (
                    data["original"],
                    data["unsharp"],
                    data["homomorphic"],
                    data["retinex"],
                    data["ground_truth"],
                )
            )
            for row in combined:
                file.write(bytes(row))


def __consolidate_data() -> (
    Generator[
        Dict[
            Literal["original", "retinex", "unsharp", "homomorphic", "ground_truth"],
            np.ndarray,
        ],
        None,
        None,
    ]
):
    # Path to intermediate images
    path_retinex = "data/intermediate_images/retinex/"
    path_unsharp = "data/intermediate_images/unsharp_masking/"
    path_homomorphic = "data/intermediate_images/homomorphic_filtering/"

    files = glob.glob("data/lol_dataset/*/low/*.png")

    for image in files:
        # read original image
        image_original = cv2.imread(image)

        # image ground truth
        image_ground_truth = cv2.imread(image.replace("low", "high"))

        # extract image id
        i = image.split("/")[-1].split(".")[0]

        # read corresponding intermediate images
        image_retinex = cv2.imread(path_retinex + str(i) + ".png")
        image_unsharp = cv2.imread(path_unsharp + str(i) + ".png")
        image_homomorphic = cv2.imread(path_homomorphic + str(i) + ".png")

        # reshape image to 2D array
        image2D_original = image_original.reshape(-1, 3)
        image2D_retinex = image_retinex.reshape(-1, 3)
        image2D_unsharp = image_unsharp.reshape(-1, 3)
        image2D_homomorphic = image_homomorphic.reshape(-1, 3)
        image2D_ground_truth = image_ground_truth.reshape(-1, 3)

        # convert to single pandas dataframe
        data: Dict[
            Literal["original", "retinex", "unsharp", "homomorphic", "ground_truth"],
            np.ndarray,
        ] = {
            "original": image2D_original,
            "retinex": image2D_retinex,
            "unsharp": image2D_unsharp,
            "homomorphic": image2D_homomorphic,
            "ground_truth": image2D_ground_truth,
        }
        yield data
