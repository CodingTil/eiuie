from abc import ABC, abstractmethod

import numpy as np


def BGR2HSI(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR to HSI.

    Parameters
    ----------
    image : np.ndarray
        Image to be converted, as BGR.

    Returns
    -------
    np.ndarray
        Converted image, as HSI.
    """

    # Normalize BGR values to [0,1]
    bgr = image.astype(np.float32) / 255.0
    blue = bgr[:, :, 0]
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]

    # Compute intensity
    I = (blue + green + red) / 3.0

    # Compute saturation
    min_val = np.minimum(np.minimum(blue, green), red)
    S = 1 - 3.0 / (blue + green + red + 1e-6) * min_val

    # Compute hue
    num = 0.5 * ((red - green) + (red - blue))
    den = np.sqrt((red - green) ** 2 + (red - blue) * (green - blue))
    theta = np.arccos(num / (den + 1e-6))
    H = theta
    H[blue > green] = 2 * np.pi - H[blue > green]
    H /= 2 * np.pi

    hsi = np.stack([H, S, I], axis=2)
    return hsi


def HSI2BGR(image: np.ndarray) -> np.ndarray:
    """
    Convert image from HSI to BGR.

    Parameters
    ----------
    image : np.ndarray
        Image to be converted, as HSI.

    Returns
    -------
    np.ndarray
        Converted image, as BGR.
    """

    H = image[:, :, 0] * 2 * np.pi
    S = image[:, :, 1]
    I = image[:, :, 2]

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # RG sector
    cond = np.logical_and(0 <= H, H < 2 * np.pi / 3)
    B[cond] = I[cond] * (1 - S[cond])
    R[cond] = I[cond] * (1 + S[cond] * np.cos(H[cond]) / np.cos(np.pi / 3 - H[cond]))
    G[cond] = 3 * I[cond] - (R[cond] + B[cond])

    # GB sector
    cond = np.logical_and(2 * np.pi / 3 <= H, H < 4 * np.pi / 3)
    H[cond] = H[cond] - 2 * np.pi / 3
    R[cond] = I[cond] * (1 - S[cond])
    G[cond] = I[cond] * (1 + S[cond] * np.cos(H[cond]) / np.cos(np.pi / 3 - H[cond]))
    B[cond] = 3 * I[cond] - (R[cond] + G[cond])

    # BR sector
    cond = np.logical_and(4 * np.pi / 3 <= H, H < 2 * np.pi)
    H[cond] = H[cond] - 4 * np.pi / 3
    G[cond] = I[cond] * (1 - S[cond])
    B[cond] = I[cond] * (1 + S[cond] * np.cos(H[cond]) / np.cos(np.pi / 3 - H[cond]))
    R[cond] = 3 * I[cond] - (G[cond] + B[cond])

    bgr = np.stack([B, G, R], axis=2)
    return (bgr * 255).astype(np.uint8)


class BaseModel(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image using the model.

        Parameters
        ----------
        image : np.ndarray
            Image to be processed, as BGR.

        Returns
        -------
        np.ndarray
            Processed image, as BGR.
        """
        ...
