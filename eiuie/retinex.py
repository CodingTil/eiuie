"""
Parts of this file are adapted from https://santhalakshminarayana.github.io/blog/retinex-image-enhancement
"""

from typing import Optional, List

import numpy as np
import cv2

import base_model as bm


def get_ksize(sigma: float) -> int:
    """
    Compute kernel size from sigma

    Parameters
    ----------
    sigma : float
        sigma value of gaussian kernel

    Returns
    -------
    int
        kernel size
    """
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0
    return int(((sigma - 0.8) / 0.15) + 2.0)


def get_gaussian_blur(
    img: np.ndarray, ksize: Optional[int] = None, sigma: float = 5.0
) -> np.ndarray:
    """
    Apply gaussian blur to image

    Parameters
    ----------
    img : np.ndarray
        image to apply gaussian blur
    ksize : Optional[int], optional
        kernel size, by default None
    sigma : float, optional
        sigma value of gaussian kernel, by default 5.0

    Returns
    -------
    np.ndarray
        gaussian blurred image
    """
    # if ksize == 0, then compute ksize from sigma
    if ksize is None:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))


def ssr(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Single-scale retinex of an image

    Parameters
    ----------
    img : np.ndarray
        image to apply single-scale retinex
    sigma : float
        sigma value of gaussian kernel

    Returns
    -------
    np.ndarray
        single-scale retinex image
    """
    # Single-scale retinex of an image
    # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
    # F = surrounding function, here Gaussian
    return np.log10(img) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)


def msr(img: np.ndarray, sigma_scales: List[float] = [15, 80, 250]) -> np.ndarray:
    """
    Multi-scale retinex of an image

    Parameters
    ----------
    img : np.ndarray
        image to apply multi-scale retinex
    sigma_scales : List[float], optional
        list of sigma values of gaussian kernel, by default [15, 80, 250]

    Returns
    -------
    np.ndarray
        multi-scale retinex image
    """
    # Multi-scale retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)

    # divide MSR by weights of each scale
    # here we use equal weights
    msr = msr / len(sigma_scales)

    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return msr


def color_balance(img: np.ndarray, low_per: float, high_per: float) -> np.ndarray:
    """
    Color balance an image, by applying contrast-stretching using histogram equalization with black and white cap

    Parameters
    ----------
    img : np.ndarray
        image to apply color balance
    low_per : float
        percentage of pixels to black-out
    high_per : float
        percentage of pixels to white-out

    Returns
    -------
    np.ndarray
        color balanced image
    """

    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    # channels of image
    ch_list = []
    if len(img.shape) == 2:
        ch_list = [img]
    else:
        ch_list = cv2.split(img)

    cs_img = []
    # for each channel, apply contrast-stretch
    for i in range(len(ch_list)):
        ch = ch_list[i]
        # cummulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if li == hi:
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array(
            [
                0 if i < li else (255 if i > hi else round((i - li) / (hi - li) * 255))
                for i in np.arange(0, 256)
            ],
            dtype="uint8",
        )
        # constrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)

    if len(cs_img) == 1:
        return np.squeeze(cs_img)
    elif len(cs_img) > 1:
        return cv2.merge(cs_img)
    raise Exception("Color balance failed")


def msrcp(
    img: np.ndarray,
    sigma_scales: List[float] = [15, 80, 250],
    low_per: float = 1,
    high_per: float = 1,
) -> np.ndarray:
    """
    Multi-scale retinex with color preservation of an image

    Parameters
    ----------
    img : np.ndarray
        image to apply multi-scale retinex with color preservation
    sigma_scales : List[float], optional
        list of sigma values of gaussian kernel, by default [15, 80, 250]
    low_per : float, optional
        percentage of pixels to black-out, by default 1
    high_per : float, optional
        percentage of pixels to white-out, by default 1

    Returns
    -------
    np.ndarray
        multi-scale retinex with color preservation image
    """
    # Multi-scale retinex with Color Preservation
    # Int(x,y) = sum(Ic(x,y))/3, c={0...k-1}, k=no.of channels
    # MSR_Int(x,y) = MSR(Int(x,y)), and apply color balance
    # B(x,y) = MAX_VALUE/max(Ic(x,y))
    # A(x,y) = max(B(x,y), MSR_Int(x,y)/Int(x,y))
    # MSRCP = A*I

    # Intensity image (Int)
    int_img = (np.sum(img, axis=2) / img.shape[2]) + 1.0
    # Multi-scale retinex of intensity image (MSR)
    msr_int = msr(int_img, sigma_scales)
    # color balance of MSR
    msr_cb = color_balance(msr_int, low_per, high_per)

    # B = MAX/max(Ic)
    B = 256.0 / (np.max(img, axis=2) + 1.0)
    # BB = stack(B, MSR/Int)
    BB = np.array([B, msr_cb / int_img])
    # A = min(BB)
    A = np.min(BB, axis=0)
    # MSRCP = A*I
    msrcp = np.clip(np.expand_dims(A, 2) * img, 0.0, 255.0)

    return msrcp.astype(np.uint8)


class Retinex(bm.BaseModel):
    def __init__(
        self,
        sigma_scales=[15, 80, 250],
        alpha=125,
        beta=46,
        G=192,
        b=-30,
        low_per=1,
        high_per=1,
    ):
        self.sigma_scales = sigma_scales
        self.alpha = alpha
        self.beta = beta
        self.G = G
        self.b = b
        self.low_per = low_per
        self.high_per = high_per

    @property
    def name(self) -> str:
        """
        Name of the model.

        Returns
        -------
        str
            Name of the model.
        """
        return "retinex"

    def process_image(self, img):
        return msrcp(img)
