import numpy as np
import cv2
from util import BGR2HSI, HSI2BGR


def filter(value, gamma_1: float = 1.0, gamma_2: float = 0.6, rho: float = 2.0):
    return gamma_1 - gamma_2 * (1 / (1 + 2.415 * np.power(value / rho, 4)))


def process_image(image: np.ndarray) -> np.ndarray:
    # Convert image to HSI space
    image = image.astype(np.float32)
    hsi = BGR2HSI(image)

    # Extract intensity channel and apply homomorphic filtering
    i = hsi[:, :, 2]
    i_log = np.log2(i + 1.0)
    i_log_fft_shifted = np.fft.fftshift(np.fft.fft2(i_log))
    i_log_fft_shifted_filtered = np.zeros_like(i_log_fft_shifted)
    for i in range(i_log_fft_shifted.shape[0]):
        for j in range(i_log_fft_shifted.shape[1]):
            i_log_fft_shifted_filtered[i, j] = i_log_fft_shifted[i, j] * filter(
                np.sqrt(
                    (i - i_log_fft_shifted.shape[0] / 2) ** 2
                    + (j - i_log_fft_shifted.shape[1] / 2) ** 2
                )
            )
    i_log_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(i_log_fft_shifted_filtered)))
    i_filtered = np.exp2(i_log_filtered) - 1.0
    # Replace intensity channel with filtered one
    hsi_filtered = hsi.copy()
    hsi_filtered[:, :, 2] = i_filtered

    # Convert image back to BGR space
    image = HSI2BGR(hsi)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    # Equalize histogram of value channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

    # Convert image back to BGR space
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image
