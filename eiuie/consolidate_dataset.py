import numpy as np
import pandas as pd
import cv2


# # Text content to be written in the file
# text_content = "This is an example ASCII text that will be written to a file."
#
# # source path and name
# data_path = "../data/"
#
# # File path and name
# file_path = "../data/dataset.txt"
#
# # Writing text content to a .txt file in ASCII encoding
# with open(file_path, 'w') as file:
#     file.write(text_content)


def image_to_pandas(source_path) -> pd.DataFrame:
    """
    Convert image to pandas dataframe.
    """

    # Path to the image file
    path_original = source_path + "lol_dataset/our485/"
    path_retinex = source_path + "intermediate/retinex/"
    path_unsharp = source_path + "intermediate/unsharp_masking/"
    path_homomorphic = source_path + "intermediate/homomorphic_filtering/"

    # Read same image in each folder
    for i in range(1, 486):
        image_original = cv2.imread(path_original + str(i) + ".png")
        image_retinex = cv2.imread(path_retinex + str(i) + ".png")
        image_unsharp = cv2.imread(path_unsharp + str(i) + ".png")
        image_homomorphic = cv2.imread(path_homomorphic + str(i) + ".png")

        # reshape image to 2D array
        image2D_original = image_original.reshape(-1, image_original.shape[-1])
        image2D_retinex = image_retinex.reshape(-1, image_retinex.shape[-1])
        image2D_unsharp = image_unsharp.reshape(-1, image_unsharp.shape[-1])
        image2D_homomorphic = image_homomorphic.reshape(-1, image_homomorphic.shape[-1])

        # convert to single pandas dataframe
        data = {
            "original": image2D_original,
            "retinex": image2D_retinex,
            "unsharp": image2D_unsharp,
            "homomorphic": image2D_homomorphic,
        }
        df = pd.DataFrame(data)
    return df
