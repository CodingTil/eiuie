from typing import List, Dict

import numpy as np
import pandas as pd
import cv2
import os
import glob

import base_model as bm
import unsharp_masking as um
import retinex as rtx
import homomorphic_filtering as hf
import fusion_model as fm


EVAL_DATASET_PREFIX = "data/eval_dataset"
EVAL_DATASET_INPUT = f"{EVAL_DATASET_PREFIX}/input"
EVAL_DATASET_OUTPUT = f"{EVAL_DATASET_PREFIX}/output"
EVAL_DATASET_RESULTS = f"{EVAL_DATASET_PREFIX}/eval_results.csv"
EVAL_DATASET_RESULTS_AVG = f"{EVAL_DATASET_PREFIX}/eval_results_avg.csv"


def rms_contrast(img: np.ndarray) -> float:
    """
    Compute the RMS Contrast score of an image.

    Parameters
    ----------
    img : np.ndarray
        The image to compute the RMS Contrast score of, as BGR.

    Returns
    -------
    float
        The RMS Contrast score of the image (between 0 and 1).
    """
    hsi = bm.BGR2HSI(img)
    i = hsi[:, :, 2]
    avg = np.mean(i)
    return np.sqrt(1 / (i.shape[0] * i.shape[1])) * np.sum(np.square(i - avg))


def discrete_entropy(img: np.ndarray) -> float:
    """
    Compute the Discrete Entropy score of an image.

    Parameters
    ----------
    img : np.ndarray
        The image to compute the Discrete Entropy score of, as BGR.

    Returns
    -------
    float
        The Discrete Entropy score of the image (between 0 and 1).
    """
    # Compute the absolute differences between adjacent pixels along rows and columns
    diff_rows = np.abs(np.diff(img, axis=0))
    diff_cols = np.abs(np.diff(img, axis=1))

    # Concatenate the differences to get a single array of differences
    diffs = np.concatenate((diff_rows, diff_cols), axis=None)

    # Get the counts of each unique difference value
    _, counts = np.unique(diffs, return_counts=True)

    # Compute the probabilities of each difference value
    probabilities = counts / np.sum(counts)

    # Compute the entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

    return float(entropy)


def batch_evaluate(
    methods: List[bm.BaseModel] = [
        um.UnsharpMasking(),
        rtx.Retinex(),
        hf.HomomorphicFiltering(),
        fm.FusionModel(),
    ],
) -> None:
    """
    Batch evaluate a list of images using a list of methods.

    Parameters
    ----------
    methods : List[bm.BaseModel], optional
        List of methods to evaluate the images with, by default [um.UnsharpMasking(), rtx.Retinex(), hf.HomomorphicFiltering(), fm.FusionModel()]
    """
    data: Dict[str, List[float | str]] = {
        "image_name": [],
        "method": [],
        "original_rms_contrast": [],
        "enhanced_rms_contrast": [],
        "original_discrete_entropy": [],
        "enhanced_discrete_entropy": [],
    }
    for image_path in glob.glob(f"{EVAL_DATASET_INPUT}/*"):
        image_path = os.path.abspath(image_path)
        print(f"\nImage {image_path}")
        image_name = image_path.split("/")[-1]
        image = cv2.imread(image_path)
        for method in methods:
            print(f"Processing with method {method.name}...")
            enhanced_image = method.process_image(image)
            # Save image
            print("Saving...")
            os.makedirs(f"{EVAL_DATASET_OUTPUT}/{method.name}", exist_ok=True)
            cv2.imwrite(
                f"{EVAL_DATASET_OUTPUT}/{method.name}/{image_name}", enhanced_image
            )
            data["image_name"].append(image_name)
            data["method"].append(method.name)
            print("RMS...")
            data["original_rms_contrast"].append(rms_contrast(image))
            data["enhanced_rms_contrast"].append(rms_contrast(enhanced_image))
            print("Discrete Entropy...")
            data["original_discrete_entropy"].append(discrete_entropy(image))
            data["enhanced_discrete_entropy"].append(discrete_entropy(enhanced_image))
    df = pd.DataFrame(data)
    df.to_csv(EVAL_DATASET_RESULTS, index=False)
    avg_df = df.copy()
    # remove image_name column
    avg_df = avg_df.drop(columns=["image_name"])
    # group by method
    avg_df = avg_df.groupby("method").mean()
    avg_df.to_csv(EVAL_DATASET_RESULTS_AVG)
