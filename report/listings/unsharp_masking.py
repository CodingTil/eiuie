import numpy as np
import cv2


def process_image(self, image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (self.ksize, self.ksize), 0)
    sharpened = cv2.addWeighted(image, 1 + self.alpha, blurred, -self.alpha, 0)
    return sharpened
