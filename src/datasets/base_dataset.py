"""Dataset base class"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import multiprocessing as mp
import logging
import torch

from src.utils.ghost_patterns import get_hadamard_matrix_cached
from src.utils.reconstruction_manifest import validate_reconstruction_manifest


class BaseSegmentationDataset(Dataset, ABC):
    def __init__(self, root_dir, bucket_size, img_size, num_classes,
                 mode='train', preload=False, augmentation=None, transform=None, num_workers=None,
                 compute_bucket=True, perm_seed=None, bucket_noise_snr_db=None,
                 require_reconstruction_manifest=False):
        self.root_dir = Path(root_dir)  # path to the dataset root directory
        self.mode = mode  # split: train, val, test, ...
        self.bucket_size = bucket_size  # length of the bucket signal
        self.img_size = img_size  # image size (square, width == height)
        self.num_classes = num_classes  # number of segmentation classes
        self.preload = preload  # whether to preload the samples into memory
        self.augmentation = augmentation  # data augmentation function
        self.transform = transform  # extra transform function
        self.compute_bucket = compute_bucket  # False = the bucket is computed in batch on the GPU by trainer/eval (speeds up carvana)
        self.perm_seed = perm_seed  # acquisition-order ablation: when not None, the rows of Φ get a fixed seeded permutation (same source as the GPU path, so the two agree)
        self.bucket_noise_snr_db = bucket_noise_snr_db  # measurement-domain noise (CPU path; None = off)

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() // 2)  # default to half the CPU cores
        self.num_workers = num_workers  # number of worker processes reading the data

        split_dir = self.root_dir / self.mode
        is_reconstruction_root = any(
            part.lower().startswith('data_recon') for part in self.root_dir.parts
        )
        if (require_reconstruction_manifest or is_reconstruction_root
                or (split_dir / '_dump_meta.json').exists()):
            validate_reconstruction_manifest(split_dir)

        self.samples = self._scan_samples()  # scan the list of sample files
        if len(self.samples) == 0:
            raise RuntimeError(f"No sample found: {self.root_dir}")  # error out when there is no sample

        # build the Hadamard matrix; with compute_bucket=False (bucket_on_gpu) no patterns are needed, which saves the large CPU matrix product and the shared memory
        self.patterns = None
        self.patterns_tensor = None
        self.use_shared_memory = False
        if self.compute_bucket:
            self.patterns = get_hadamard_matrix_cached(self.img_size * self.img_size, self.bucket_size, self.perm_seed)
            if self.num_workers > 0:
                try:
                    self.patterns_tensor = torch.from_numpy(self.patterns).share_memory_()
                    self.use_shared_memory = True
                except Exception:
                    pass

        self.cache = None  # sample cache, holds the images and masks when preloading
        if self.preload:
            logging.info(f"Preloading {len(self.samples)} {self.mode} samples...")  # log the preload notice
            self._preload_samples()  # run the preload

    def _preload_samples(self):
        self.cache = []
        for i in range(len(self.samples)):
            try:
                img_pil, msk_pil = self._load_pair(i)  # load the image and mask as PIL objects
                img_pil = img_pil.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)  # resize the image
                msk_pil = msk_pil.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)  # resize the mask
                self.cache.append((img_pil, msk_pil))  # append to the cache list
            except Exception as e:
                logging.warning(f"Failed to preload sample {i}: {e}")  # preload failure log
                self.cache.append(None)  # store None as a placeholder on failure

    @abstractmethod
    def _scan_samples(self):
        pass  # implemented by the subclass: scan the list of sample files

    @abstractmethod
    def _load_pair(self, idx):
        pass  # implemented by the subclass: load one image-mask pair

    def __len__(self):
        return len(self.samples)  # return the total number of samples

    def __getitem__(self, idx):
        # load from the cache first (when preloaded and the cache entry is valid)
        if self.cache is not None and self.cache[idx] is not None:
            img_pil, msk_pil = self.cache[idx]
            img_pil = img_pil.copy()  # copy, so later edits cannot reach the cache
            msk_pil = msk_pil.copy()
        else:
            img_pil, msk_pil = self._load_pair(idx)  # reload the image and mask
            img_pil = img_pil.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            msk_pil = msk_pil.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        # apply data augmentation in train mode
        if self.augmentation is not None and self.mode == 'train':
            img_pil, msk_pil = self.augmentation(img_pil, msk_pil)

        # convert to numpy arrays, normalize the image pixels to [0,1]
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        mask_np = np.array(msk_pil)

        # if the mask has several channels, keep the first one
        if mask_np.ndim > 2:
            mask_np = mask_np[..., 0]

        # handle the labels: binarize for a single class, clip to the class range for several classes
        if self.num_classes == 1:
            mask_np = (mask_np > 127).astype(np.int64)
        else:
            unique_values = np.unique(mask_np)
            if unique_values.max() > self.num_classes - 1:
                logging.warning(f"Sample {idx} has labels out of range")  # warn when a label value exceeds the number of classes
                mask_np = np.clip(mask_np, 0, self.num_classes - 1).astype(np.int64)
            else:
                mask_np = mask_np.astype(np.int64)

        # build the sample: image/mask/index are always there; the bucket is computed on the CPU by default, and with compute_bucket=False the GPU side computes it (skipped here)
        sample = {
            'image': img_np[None, ...].astype(np.float32),  # (1,H,W) float32, [0,1]
            'mask': mask_np.astype(np.int64),
            'index': idx,
        }
        if self.compute_bucket:
            if self.use_shared_memory and self.patterns_tensor is not None:
                bucket_raw = torch.matmul(self.patterns_tensor, torch.from_numpy(img_np.reshape(-1))).numpy()
            else:
                bucket_raw = self.patterns @ img_np.reshape(-1)
            if self.bucket_noise_snr_db is not None:
                # measurement-domain Gaussian noise (the additive measurement-noise term), calibrated on the signal standard deviation: sigma = std(ref)*10^(-SNR/20)
                # NOTE: this CPU branch is not the path the shipped configs use. They all set
                # bucket_on_gpu=true, so training noise is injected by src/utils/bucket.py instead, and
                # `bucket_noise_ref` is never assigned by any config or script — the reference below is
                # therefore always the full bucket. Assign the attribute yourself to get the 'ac' behaviour.
                ref = bucket_raw[1:] if getattr(self, 'bucket_noise_ref', 'full') == 'ac' else bucket_raw
                sigma = float(ref.std()) * (10.0 ** (-float(self.bucket_noise_snr_db) / 20.0))
                bucket_raw = (bucket_raw + np.random.normal(0.0, sigma, bucket_raw.shape)).astype(np.float32)
            bucket_norm = (bucket_raw - bucket_raw.min()) / (bucket_raw.max() - bucket_raw.min() + 1e-8)
            sample['bucket'] = bucket_norm.astype(np.float32)
            sample['bucket_raw'] = bucket_raw.astype(np.float32)

        # apply the extra transform to the sample dict if there is one
        if self.transform:
            sample = self.transform(sample)

        return sample

    def validate(self):
        errors = []
        # only validate the first 10 samples, so this stays cheap
        for i in range(min(10, len(self.samples))):
            try:
                sample = self[i]  # read the sample
                assert 'image' in sample
                assert 'mask' in sample
                if self.compute_bucket:
                    assert 'bucket' in sample
                    assert 'bucket_raw' in sample
            except Exception as e:
                errors.append(f"Sample {i}: {e}")  # record the exception

        if errors:
            for error in errors:
                logging.error(error)  # emit the error log
            return False
        return True
