from __future__ import annotations  # enable the newer type annotation syntax
from dataclasses import dataclass  # dataclass keeps the class definition short
from typing import Tuple, Optional, Dict, Any  # type annotations
import random  # random module
import numpy as np  # numpy for numerical work
from PIL import Image, ImageEnhance  # PIL image handling and enhancement modules


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)  # clamp the array values to the range 0 to 1


@dataclass
class PairAugment:
    img_size: int  # target size of the output image, same width and height
    horizontal_flip: float = 0.0  # probability of a horizontal flip
    vertical_flip: float = 0.0  # probability of a vertical flip
    rotate_degree: float = 0.0  # maximum rotation angle (plus/minus range)
    brightness: float = 0.0  # brightness adjustment magnitude
    contrast: float = 0.0  # contrast adjustment magnitude
    gaussian_noise_std: float = 0.0  # standard deviation of the Gaussian noise
    random_crop_scale: Optional[Tuple[float, float]] = None  # range of the random crop ratio
    enable: bool = True  # whether augmentation is enabled

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not self.enable:
            # with augmentation disabled, just resize so the output size stays consistent
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            return img, mask

        # random crop: crop a region scaled off the shorter side, then resize back to the target size
        if self.random_crop_scale and isinstance(self.random_crop_scale, (list, tuple)) and len(self.random_crop_scale) == 2:
            low, high = float(self.random_crop_scale[0]), float(self.random_crop_scale[1])
            low = max(0.1, min(1.0, low))  # keep the minimum ratio no lower than 0.1
            high = max(low, min(1.0, high))  # keep the maximum ratio no higher than 1.0 and no lower than low
            scale = random.uniform(low, high)  # draw the crop ratio at random
            if scale < 0.999:  # only crop when a crop is actually needed
                w, h = img.size
                cw, ch = int(w * scale), int(h * scale)  # width and height after cropping
                if cw >= 1 and ch >= 1 and cw <= w and ch <= h:
                    x0 = random.randint(0, w - cw)  # random x coordinate of the crop origin
                    y0 = random.randint(0, h - ch)  # random y coordinate of the crop origin
                    img = img.crop((x0, y0, x0 + cw, y0 + ch))  # crop the image
                    mask = mask.crop((x0, y0, x0 + cw, y0 + ch))  # crop the mask

        # random horizontal flip, probability controlled by horizontal_flip
        if self.horizontal_flip > 0 and random.random() < self.horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random vertical flip, probability controlled by vertical_flip
        if self.vertical_flip > 0 and random.random() < self.vertical_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # random rotation over the range [-rotate_degree, rotate_degree]
        if self.rotate_degree and self.rotate_degree > 0:
            deg = random.uniform(-float(self.rotate_degree), float(self.rotate_degree))
            img = img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=0)  # rotate the image with bilinear interpolation
            mask = mask.rotate(deg, resample=Image.NEAREST, expand=False, fillcolor=0)  # rotate the mask with nearest neighbour to avoid mixing classes

        # adjust the brightness, range of increase/decrease set by brightness
        if self.brightness and self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # adjust the contrast, range of increase/decrease set by contrast
        if self.contrast and self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # add Gaussian noise, applied to the image only
        if self.gaussian_noise_std and self.gaussian_noise_std > 0:
            arr = np.asarray(img).astype(np.float32) / 255.0  # normalize to [0,1]
            noise = np.random.normal(0.0, float(self.gaussian_noise_std), size=arr.shape).astype(np.float32)  # draw the Gaussian noise
            arr = _clamp01(arr + noise)  # clip back to [0,1] after adding the noise
            img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode=img.mode)  # convert back to a PIL image

        # make sure the output image has the target size
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if mask.size != (self.img_size, self.img_size):
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        return img, mask  # return the augmented image and mask


def build_transforms(cfg: Dict[str, Any], img_size: int):
    cfg = cfg or {}  # fall back to an empty dict when the config is empty
    enable = bool(cfg.get('enable', False))  # whether augmentation is enabled
    return PairAugment(
        img_size=img_size,
        horizontal_flip=float(cfg.get('horizontal_flip', 0.0) or 0.0),  # probability of a horizontal flip
        vertical_flip=float(cfg.get('vertical_flip', 0.0) or 0.0),  # probability of a vertical flip
        rotate_degree=float(cfg.get('rotate_degree', 0.0) or 0.0),  # maximum rotation angle
        brightness=float(cfg.get('brightness', 0.0) or 0.0),  # brightness adjustment magnitude
        contrast=float(cfg.get('contrast', 0.0) or 0.0),  # contrast adjustment magnitude
        gaussian_noise_std=float(cfg.get('gaussian_noise_std', 0.0) or 0.0),  # standard deviation of the Gaussian noise
        random_crop_scale=tuple(cfg.get('random_crop_scale', [])) if cfg.get('random_crop_scale') else None,  # range of the random crop ratio
        enable=enable  # whether augmentation is enabled
    )