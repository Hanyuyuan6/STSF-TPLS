from .base_dataset import BaseSegmentationDataset  # the base class
from pathlib import Path  # Path, which makes the path handling easy
from PIL import Image  # PIL, for loading and processing images
import logging  # the logging module, for the log records


class FolderSegDataset(BaseSegmentationDataset):
    """
    Generic folder-structured dataset, for the standard images/masks directory layout
    """

    def _scan_samples(self):
        split_dir = self.root_dir / self.mode  # directory of the current split (train/val)
        img_dir = split_dir / 'images'  # the image directory
        msk_dir = split_dir / 'masks'  # the mask directory

        if not img_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {img_dir}")  # error out when the image directory is missing
        if not msk_dir.exists():
            raise RuntimeError(f"Mask directory does not exist: {msk_dir}")  # error out when the mask directory is missing

        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']  # the supported image formats
        mask_extensions = ['.png', '.gif', '.jpg', '.jpeg', '.tif', '.tiff']  # the supported mask formats

        img_files = []  # collects every image path
        for ext in image_extensions:
            img_files.extend(img_dir.glob(f'*{ext}'))  # scan the lower-case extensions
            img_files.extend(img_dir.glob(f'*{ext.upper()}'))  # scan the upper-case extensions

        img_files = sorted(set(img_files))  # deduplicate and sort the image file list

        pairs = []  # collects the image-mask pairs
        missing_masks = []  # names of the image files whose mask is missing

        for img_path in img_files:
            mask_found = False  # tracks whether the matching mask was found

            for ext in mask_extensions:
                # rule 1: the mask file has the same name as the image, only the extension differs
                msk_path = msk_dir / (img_path.stem + ext)
                if msk_path.exists():
                    pairs.append((img_path, msk_path))  # add the pair
                    mask_found = True
                    break

                # rule 2: the mask file name carries a "_mask" suffix
                msk_path = msk_dir / (img_path.stem + '_mask' + ext)
                if msk_path.exists():
                    pairs.append((img_path, msk_path))  # add the pair
                    mask_found = True
                    break

            if not mask_found:
                missing_masks.append(img_path.name)  # record the image whose mask is missing

        if missing_masks:
            logging.warning(f"Found {len(missing_masks)} images with no matching mask")  # print the missing-mask warning
            if len(missing_masks) <= 10:
                for name in missing_masks:
                    logging.debug(f"  missing mask: {name}")  # print the missing mask file names one by one (debug level)

        if len(pairs) == 0:
            raise RuntimeError(f"No valid image-mask pair found in {split_dir}")  # error out when no pair is valid

        logging.info(f"Dataset {self.root_dir.name}: found {len(pairs)} {self.mode} pairs")  # record how many samples were found
        return pairs  # return the list of image-mask pairs

    def _load_pair(self, idx):
        img_path, msk_path = self.samples[idx]  # image and mask path for this index

        try:
            img = Image.open(img_path).convert('L')  # load the image, converted to grayscale

            msk = Image.open(msk_path)  # load the mask image
            if msk.mode != 'L':
                msk = msk.convert('L')  # convert to grayscale unless it already is

            return img, msk  # return the image and mask as PIL objects

        except Exception as e:
            logging.error(f"Load failed - image: {img_path}, mask: {msk_path}")  # load failure log
            raise e  # re-raise the exception