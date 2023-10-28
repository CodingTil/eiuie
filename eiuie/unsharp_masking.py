import numpy as np
import cv2

import base_model as bm


class UnsharpMasking(bm.BaseModel):
    """Unsharp Masking"""

    ksize: int
    sigma: float

    def __init__(self, ksize: int = 33, sigma: float = 20.0):
        """
        Parameters
        ----------
        ksize : int, optional
            Kernel size, by default 33
        sigma : float, optional
            Gaussian kernel standard deviation, by default 20.0
        """
        self.ksize = ksize
        self.sigma = sigma

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
        blurred = cv2.GaussianBlur(image, (self.ksize, self.ksize), self.sigma)
        sharpened = cv2.addWeighted(image, 2, blurred, -1, 0)
        return sharpened
