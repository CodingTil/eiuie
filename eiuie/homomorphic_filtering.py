import numpy as np
import cv2

import base_model as bm


class HomomorphicFiltering(bm.BaseModel):
    """Homomorphic Filtering"""

    gamma_1: float
    gamma_2: float
    rho: float

    def __init__(
        self, gamma_high: float = 1.0, gamma_low: float = 0.4, rho: float = 2.0
    ):
        """
        Parameters
        ----------
        gamma_high : float, optional
            High gamma value, by default 1.0
        gamma_low : float, optional
            Low gamma value, by default 0.4
        rho : float, optional
            Rho value, by default 2.0
        """
        self.gamma_1 = gamma_high
        self.gamma_2 = gamma_high - gamma_low
        self.rho = rho

    @property
    def name(self) -> str:
        """
        Name of the model.

        Returns
        -------
        str
            Name of the model.
        """
        return "homomorphic_filtering"

    def filter(self, value):
        return self.gamma_1 - self.gamma_2 * (
            1 / (1 + 2.415 * np.power(value / self.rho, 4))
        )

    def _process_image(self, hsi: np.ndarray) -> np.ndarray:
        """
        Process image using the model.

        Parameters
        ----------
        image : np.ndarray
            Image to be processed, as HSI.

        Returns
        -------
        np.ndarray
            Processed image, as HSI.
        """
        i = hsi[:, :, 2]
        i_log = np.log2(i + 1.0)
        i_log_fft_shifted = np.fft.fftshift(np.fft.fft2(i_log))
        i_log_fft_shifted_filtered = np.zeros_like(i_log_fft_shifted)
        for i in range(i_log_fft_shifted.shape[0]):
            for j in range(i_log_fft_shifted.shape[1]):
                i_log_fft_shifted_filtered[i, j] = i_log_fft_shifted[
                    i, j
                ] * self.filter(
                    np.sqrt(
                        (i - i_log_fft_shifted.shape[0] / 2) ** 2
                        + (j - i_log_fft_shifted.shape[1] / 2) ** 2
                    )
                )
        i_log_filtered = np.real(
            np.fft.ifft2(np.fft.ifftshift(i_log_fft_shifted_filtered))
        )
        i_filtered = np.exp2(i_log_filtered) - 1.0
        # between 0 and 1
        if np.max(i_filtered) > 1.0 or np.min(i_filtered) < 0.0:
            i_filtered = (i_filtered - np.min(i_filtered)) / (
                np.max(i_filtered) - np.min(i_filtered)
            )
        hsi_filtered = hsi.copy()
        hsi_filtered[:, :, 2] = i_filtered
        return hsi_filtered

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
        image = image.astype(np.float32)
        hsi = bm.BGR2HSI(image)
        hsi = self._process_image(hsi)
        image = bm.HSI2BGR(hsi)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image
