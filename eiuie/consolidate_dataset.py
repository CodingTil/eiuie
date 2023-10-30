import numpy as np
import pandas as pd
import cv2
import glob


def consolidate_data(image_files, source_path) -> pd.DataFrame:
    """
    consolidate data
    """
    # Path to intermediate images
    path_retinex = source_path + "intermediate_images/retinex/"
    path_unsharp = source_path + "intermediate_images/unsharp_masking/"
    path_homomorphic = source_path + "intermediate_images/homomorphic_filtering/"

    list_of_dicts = []
    for image in image_files:
        # read original image
        image_original = cv2.imread(image)

        # extract image id
        i = image.split("/")[-1].split(".")[0]

        # read corresponding intermediate images
        image_retinex = cv2.imread(path_retinex + str(i) + ".png")
        image_unsharp = cv2.imread(path_unsharp + str(i) + ".png")
        image_homomorphic = cv2.imread(path_homomorphic + str(i) + ".png")

        # reshape image to 2D array
        image2D_original = image_original.reshape((image_original.shape[0]*image_original.shape[1], 3))
        image2D_retinex = image_retinex.reshape((image_retinex.shape[0]*image_retinex.shape[1], 3))
        image2D_unsharp = image_unsharp.reshape((image_unsharp.shape[0]*image_unsharp.shape[1], 3))
        image2D_homomorphic = image_homomorphic.reshape((image_homomorphic.shape[0]*image_homomorphic.shape[1], 3))

        # convert to single pandas dataframe
        data = {
            "original": image2D_original,
            "retinex": image2D_retinex,
            "unsharp": image2D_unsharp,
            "homomorphic": image2D_homomorphic,
        }
        list_of_dicts.append(data)
    return list_of_dicts

def write_to_tsv(dataset, source_path):
    """
    Write dataset to tsv file.
    """

    # write to csv file
    with open(source_path + "dataset.tsv", "w") as file:
        for data in dataset:
            # write data to tsv file in the following format: original, unsharp, homomorphic, retinex
            for i in range(len(data["original"])):
                line = [data['original'][i, 0], data['original'][i, 1], data['original'][i, 2],
                        data['unsharp'][i, 0], data['unsharp'][i, 1], data['unsharp'][i, 2],
                        data['homomorphic'][i, 0], data['homomorphic'][i, 1], data['homomorphic'][i, 2],
                        data['retinex'][i, 0], data['retinex'][i, 1], data['retinex'][i, 2]]

                # write line to file
                line_str = '\t'.join(map(str, line))  # Convert vector elements to strings and join with tabs
                file.write(line_str + '\n')  # Writing the vector as a single line
    return 0


# source path
source_path = "../data/"

# consolidate dataset in pandas dataframe
glob_pattern = source_path + "lol_dataset/our485/low/*.png"
image_files = glob.glob(glob_pattern)
dataset = consolidate_data(image_files, source_path)

# write dataset to tsv file
write_to_tsv(dataset, source_path)



