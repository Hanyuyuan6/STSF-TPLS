from .folder_dataset import FolderSegDataset
from .base_dataset import BaseSegmentationDataset
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import cv2


def canonical_gray_map(msk_array, num_classes):
    """WBC mask gray level -> class index: snap every pixel to the nearest canonical gray level.

    WBC masks encode the classes as fixed, evenly spaced gray levels (3 classes -> 0/128/255 = background/cytoplasm/nucleus). A few masks carry stray
    annotation-noise gray levels; snapping to the nearest canonical level makes the class depend on **the gray value itself** rather than on the "rank"
    of the levels that happen to occur in that image -- otherwise a noise value below 128 usurps the real cytoplasm/nucleus rank and mislabels the
    foreground of the whole image (Dataset2/034 is exactly that case; rank-of-unique was the original form of this bug).

    This is a module-level function rather than inline code so that `test/test_anchors.py` can **import it and actually test it**:
    that test used to re-implement the same logic inside its own body, so reverting the source to rank-of-unique still came out all green
    (mutation-proven 2026-07-17). Do not inline it back.

    Args:
        msk_array: mask gray-level array (H, W), any integer dtype.
        num_classes: number of classes; the canonical gray levels are linspace(0, 255, num_classes) rounded to integers.
    Returns:
        (H, W) uint8 array of class indices, valued in [0, num_classes-1].
    """
    msk_array = np.asarray(msk_array).astype(np.int16)
    levels = np.rint(np.linspace(0, 255, num_classes)).astype(np.int16)
    return np.abs(msk_array[..., None] - levels).argmin(-1).astype(np.uint8)


class CarvanaDataset(FolderSegDataset):
    """Carvana car segmentation dataset, handles several image and mask formats"""

    def _scan_samples(self):
        split_dir = self.root_dir / self.mode
        img_dir = split_dir / 'images'
        default_msk_dir = split_dir / 'masks'

        if not img_dir.exists() and split_dir.exists():
            img_dir = split_dir

        candidate_mask_dirs = [
            default_msk_dir,
            self.root_dir / f'{self.mode}_masks',
            self.root_dir / 'masks',
            split_dir / 'masks',
            img_dir.parent / 'masks'
        ]
        candidate_mask_dirs = [d for d in candidate_mask_dirs if d is not None]

        if not img_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {img_dir}")

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        mask_extensions = ['.gif', '.png', '.jpg', '.jpeg', '.tif', '.tiff']

        img_files = []
        for ext in image_extensions:
            img_files.extend(img_dir.glob(f'*{ext}'))
            img_files.extend(img_dir.glob(f'*{ext.upper()}'))
        img_files = sorted(set(img_files))

        pairs = []
        missing = []

        for img_path in img_files:
            stem = img_path.stem
            found = False
            for mdir in candidate_mask_dirs:
                if not mdir.exists():
                    continue
                for name in [f'{stem}_mask', stem]:
                    for ext in mask_extensions:
                        msk_path = mdir / f'{name}{ext}'
                        if msk_path.exists():
                            pairs.append((img_path, msk_path))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                missing.append(img_path.name)

        if missing:
            logging.warning(f"Carvana: {len(missing)} images have no mask")

        if len(pairs) == 0:
            raise RuntimeError(f"No Carvana data pair found in {split_dir}")

        logging.info(f"Carvana dataset: found {len(pairs)} {self.mode} pairs")
        return pairs

    def _load_pair(self, idx):
        img_path, msk_path = self.samples[idx]
        img = Image.open(img_path).convert('L')

        msk = Image.open(msk_path)
        if msk.mode != 'L':
            msk = msk.convert('L')
        msk_array = np.array(msk)
        msk_array = ((msk_array > 127) * 255).astype(np.uint8)
        msk = Image.fromarray(msk_array)
        return img, msk


class WBCDataset(FolderSegDataset):
    """White blood cell segmentation, supports CLAHE enhancement"""

    def __init__(self, *args, use_clahe=True, **kwargs):
        self.use_clahe = use_clahe
        super().__init__(*args, **kwargs)

    def _load_pair(self, idx):
        img_path, msk_path = self.samples[idx]
        img = Image.open(img_path)

        if img.mode == 'RGB':
            img_array = np.array(img)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            img_gray = lab[:, :, 0]
            if self.use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_gray = clahe.apply(img_gray)
            img = Image.fromarray(img_gray)
        else:
            img = img.convert('L')

        msk = Image.open(msk_path).convert('L')

        msk = Image.fromarray(canonical_gray_map(np.array(msk), self.num_classes))
        return img, msk


class USNerveDataset(FolderSegDataset):
    """Ultrasound nerve segmentation, supports bilateral-filter denoising"""

    def __init__(self, *args, denoise=True, **kwargs):
        self.denoise = denoise
        super().__init__(*args, **kwargs)

    def _load_pair(self, idx):
        img_path, msk_path = self.samples[idx]
        img = Image.open(img_path).convert('L')

        if self.denoise:
            img_array = np.array(img)
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            img_array = cv2.equalizeHist(img_array)
            img = Image.fromarray(img_array)

        msk = Image.open(msk_path).convert('L')
        return img, msk


class VOCDataset(FolderSegDataset):
    """Pascal VOC2012 segmentation dataset"""

    def _load_pair(self, idx):
        img_path, msk_path = self.samples[idx]
        img = Image.open(img_path).convert('L')

        msk = Image.open(msk_path)
        m = np.array(msk).astype(np.uint8)
        m[m == 255] = 0

        if self.num_classes == 1:
            m = (m > 0).astype(np.uint8) * 255

        msk_l = Image.fromarray(m, mode='L')
        return img, msk_l


class MNISTSegDataset(BaseSegmentationDataset):
    """MNIST as binary segmentation: image = the upscaled handwritten digit, mask = the digit foreground (pixels > 0).
    train/val come from the official train set (val = a deterministic slice of the last 10%), test from the official test set. torchvision downloads it, no prepare step needed."""

    def _scan_samples(self):
        import torchvision
        is_train_pool = self.mode in ('train', 'val')
        self._mnist = torchvision.datasets.MNIST(
            root=str(self.root_dir), train=is_train_pool, download=True)
        n = len(self._mnist)
        if self.mode == 'train':
            pool, cap = list(range(0, int(n * 0.9))), 5000
        elif self.mode == 'val':
            pool, cap = list(range(int(n * 0.9), n)), 1000
        else:  # test
            pool, cap = list(range(n)), 2000
        # deterministic evenly spaced subsampling (keeps the digit class distribution balanced), holding the size at a scale that can be rerun over several seeds
        if len(pool) > cap:
            sel = np.linspace(0, len(pool) - 1, cap).round().astype(int)
            pool = [pool[i] for i in sel]
        return pool

    def _load_pair(self, idx):
        img, _ = self._mnist[self.samples[idx]]              # PIL 'L' 28x28
        img = img.convert('L')
        mask = Image.fromarray((np.array(img) > 0).astype(np.uint8) * 255, mode='L')
        return img, mask