import numpy as np
import cv2

import base_model as bm


class UnsharpMasking(bm.BaseModel):
    """Unsharp Masking"""

    ksize: int
    alpha: float

    def __init__(self, ksize: int = 91, alpha: float = 0.6):
        """
        Parameters
        ----------
        ksize : int, optional
            Kernel size, by default 91
        alpha : float, optional
            Alpha value, by default 0.6
        """
        self.ksize = ksize
        self.alpha = alpha

    @property
    def name(self) -> str:
        """
        Name of the model.

        Returns
        -------
        str
            Name of the model.
        """
        return "unsharp_masking"

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
        blurred = cv2.GaussianBlur(image, (self.ksize, self.ksize), 0)
        sharpened = cv2.addWeighted(image, 1 + self.alpha, blurred, -self.alpha, 0)
        return sharpened
