import argparse
import hashlib
import logging
from pathlib import Path
import shutil
import random
from typing import Tuple, List
import json
import tempfile
import zipfile
import tarfile
import gzip
import stat
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SegmentationDatasetPreparer:
    """
    Segmentation dataset preparation script: extracts archives automatically, lays out the data structure, supports VOC.
    """
    WBC_PROTOCOLS = ('paper-legacy-v1', 'full-v2')

    def __init__(self, data_root: str = './data'):
        self.data_root = Path(data_root)  # path of the data root directory
        self.data_root.mkdir(parents=True, exist_ok=True)  # make sure the data root exists

        # per-dataset config: raw data directory, image and mask matching rules, validation ratio
        self.dataset_configs = {
            'carvana': {
                'raw_dir': 'raw',
                'image_pattern': '*.jpg',
                'mask_pattern': '*_mask.gif',
                'val_ratio': 0.1, 'test_ratio': 0.1,
                'group_by': 'carid',   # all 16 views of one car go into the same split, to keep near-duplicates from leaking
            },
            'wbc': {
                'raw_dir': 'raw',
                'image_pattern': '*.bmp',
                'mask_pattern': '*.png',
                'val_ratio': 0.15, 'test_ratio': 0.15,
                'group_by': None,      # every image is an independent cell crop, so split per image
            },
            'us_nerve': {
                'raw_dir': 'raw',
                'image_pattern': '*.tif',
                'mask_pattern': '*.tif',
                'val_ratio': 0.1, 'test_ratio': 0.1,
                'group_by': None,
            },
            'voc': {
                'raw_dir': 'raw',
                'val_ratio': 0.1  # VOC follows the official train/val split, so this ratio is unused
            }
        }

    # Extract archives of the various formats into the given directory
    def _extract_archive(self, archive_path: Path, extract_to: Path):
        extract_to.mkdir(parents=True, exist_ok=True)  # create the extraction target directory
        logging.info(f"Extracting: {archive_path} -> {extract_to}")
        if archive_path.suffix == '.zip':  # zip format
            with zipfile.ZipFile(archive_path, 'r') as zf:
                dest = extract_to.resolve()
                for member in tqdm(zf.infolist(), desc="Extracting files"):  # show a progress bar
                    target = (extract_to / member.filename).resolve()
                    if target != dest and dest not in target.parents:
                        raise ValueError(
                            f"Archive member lands outside the extraction root, extraction refused: "
                            f"{member.filename}"
                        )
                    if stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError(
                            f"Archive symbolic link is not allowed: {member.filename}"
                        )
                    zf.extract(member, extract_to)
        elif archive_path.suffix in ['.tar', '.tgz']:  # tar or tgz format
            with tarfile.open(archive_path, 'r:*') as tf:
                dest = extract_to.resolve()
                for member in tqdm(tf.getmembers(), desc="Extracting files"):
                    # CVE-2007-4559 path-traversal guard: the tarfile of Python 3.10 has no default filter,
                    # so check member by member that the landing point is still inside dest (rejecting ../ escapes and absolute-path members).
                    target = (extract_to / member.name).resolve()
                    if target != dest and dest not in target.parents:
                        raise ValueError(f"Archive member lands outside the extraction root (path traversal), extraction refused: {member.name}")
                    if not (member.isdir() or member.isreg()):
                        raise ValueError(
                            f"Archive non-regular member is not allowed: {member.name}"
                        )
                    tf.extract(member, extract_to)
        elif archive_path.suffix == '.gz':  # gz format
            output_file = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as gz_file:
                with open(output_file, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    # Automatically extract every supported archive in the raw directory
    def _auto_extract_raw(self, raw_dir: Path):
        for f in raw_dir.iterdir():
            if f.suffix in ['.zip', '.tar', '.tgz', '.gz']:
                self._extract_archive(f, raw_dir)

    # Tell whether the dataset is already prepared (do the train, val and test image folders exist, and is train non-empty)
    def _check_dataset_ready(self, dataset_dir: Path) -> bool:
        train_imgs = dataset_dir / 'train' / 'images'
        val_imgs = dataset_dir / 'val' / 'images'
        test_imgs = dataset_dir / 'test' / 'images'
        return (train_imgs.exists() and val_imgs.exists() and test_imgs.exists()
                and len(list(train_imgs.iterdir())) > 0)

    # Create the standard dataset directory structure
    def _create_directory_structure(self, dataset_dir: Path):
        for sub in ['raw', 'train/images', 'train/masks', 'val/images', 'val/masks',
                    'test/images', 'test/masks']:
            (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy image/mask pairs without ever allowing two sources to share a destination filename.
    def _copy_pairs_to_split(self, pairs: List[Tuple[Path, Path]], split_dir: Path,
                             output_names=None):
        img_dir = split_dir / 'images'
        mask_dir = split_dir / 'masks'
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        planned = []
        seen_images, seen_masks = set(), set()
        for img_path, mask_path in pairs:
            if output_names is None:
                img_name = img_path.name
                mask_name = img_path.stem + mask_path.suffix
            else:
                img_name, mask_name = output_names(img_path, mask_path)
            if Path(img_name).name != img_name or Path(mask_name).name != mask_name:
                raise ValueError("prepared output names must be plain filenames")
            image_key, mask_key = img_name.casefold(), mask_name.casefold()
            if image_key in seen_images or mask_key in seen_masks:
                raise ValueError(
                    f"destination filename collision in {split_dir.name}: "
                    f"{img_name} / {mask_name}"
                )
            seen_images.add(image_key)
            seen_masks.add(mask_key)
            planned.append((img_path, mask_path, img_name, mask_name))

        for img_path, mask_path, img_name, mask_name in tqdm(
                planned, desc=f"Copying to {split_dir.name}"):
            shutil.copy2(img_path, img_dir / img_name)
            shutil.copy2(mask_path, mask_dir / mask_name)
        return planned

    # Save the dataset info to a json file
    def _save_dataset_info(self, dataset_dir: Path, info: dict):
        self._save_json(dataset_dir / 'dataset_info.json', info)

    @staticmethod
    def _save_json(path: Path, payload: dict):
        """Write metadata atomically so an interrupted run cannot leave truncated JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        with open(temporary, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write('\n')
        temporary.replace(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _wbc_protocol_ready(self, dataset_dir: Path, protocol_id: str) -> bool:
        """Require explicit protocol metadata and counts that match files on disk."""
        info_path = dataset_dir / 'dataset_info.json'
        manifest_path = dataset_dir / 'wbc_split_manifest.json'
        if not info_path.exists() or not manifest_path.exists():
            return False
        try:
            info = json.loads(info_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if info.get('protocol_id') != protocol_id:
            return False
        for split in ('train', 'val', 'test'):
            images = [p for p in (dataset_dir / split / 'images').iterdir() if p.is_file()]
            masks = [p for p in (dataset_dir / split / 'masks').iterdir() if p.is_file()]
            if len(images) != len(masks) or info.get(f'{split}_samples') != len(images):
                return False
        return True

    # Disjoint three-way group split into train/val/test (keeps near-duplicates from leaking). group_by='carid' groups by car number, None splits per image.
    def _grouped_three_way_split(self, pairs, val_ratio=0.1, test_ratio=0.1, group_by=None, seed=42):
        def gkey(img_path, idx):
            if group_by == 'carid':
                return img_path.stem.rsplit('_', 1)[0]
            return f'__img_{idx}'
        groups = {}
        for idx, (img, msk) in enumerate(pairs):
            groups.setdefault(gkey(img, idx), []).append((img, msk))
        gids = sorted(groups.keys())
        random.seed(seed)
        random.shuffle(gids)
        n_total = len(pairs)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        buckets = {'train': [], 'val': [], 'test': []}
        for gid in gids:
            grp = groups[gid]
            if test_ratio > 0 and len(buckets['test']) < n_test:
                buckets['test'] += grp
            elif len(buckets['val']) < n_val:
                buckets['val'] += grp
            else:
                buckets['train'] += grp
        if group_by:  # only assert disjointness when the grouping is real
            ks = lambda items: {p[0].stem.rsplit('_', 1)[0] for p in items}
            a, b, c = ks(buckets['train']), ks(buckets['val']), ks(buckets['test'])
            assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c), "groups leaked across the splits"
        logging.info(f"Three-way split (group_by={group_by}): train {len(buckets['train'])} / "
                     f"val {len(buckets['val'])} / test {len(buckets['test'])}")
        return buckets['train'], buckets['val'], buckets['test']
#############################################################################################################################################
    # Main flow: prepare the requested dataset
    def prepare_dataset(self, dataset_name: str, force: bool = False,
                        wbc_protocol: str = 'paper-legacy-v1') -> bool:
        if dataset_name not in self.dataset_configs:
            logging.error(f"Unknown dataset: {dataset_name}")
            return False

        dataset_dir = self.data_root / dataset_name
        config = self.dataset_configs[dataset_name]

        if wbc_protocol not in self.WBC_PROTOCOLS:
            logging.error(f"Unknown WBC protocol: {wbc_protocol}")
            return False

        if self._check_dataset_ready(dataset_dir) and not force:
            if dataset_name != 'wbc' or self._wbc_protocol_ready(dataset_dir, wbc_protocol):
                logging.info(f"{dataset_name} dataset is already prepared")
                return True
            logging.error(
                "Existing WBC data has missing, stale, or different protocol metadata. "
                "Refusing to guess; rerun with --force and the intended --wbc_protocol."
            )
            return False

        self._create_directory_structure(dataset_dir)

        raw_dir = dataset_dir / config['raw_dir']
        self._auto_extract_raw(raw_dir)  # automatically extract the archives in the raw directory

        # dispatch to the preparation function of the dataset at hand
        if dataset_name == 'wbc':
            return self._prepare_wbc(dataset_dir, wbc_protocol)
        elif dataset_name == 'us_nerve':
            return self._prepare_us_nerve(dataset_dir)
        elif dataset_name == 'voc':
            return self._prepare_voc(dataset_dir)
        elif dataset_name == 'carvana':
            return self._prepare_generic(dataset_dir, config)
        else:
            return self._prepare_generic(dataset_dir, config)

    @staticmethod
    def _resolve_wbc_legacy_collisions(pairs: List[Tuple[Path, Path]]):
        """Make the historical last-write-wins selection explicit before any copy occurs."""
        winners = {}
        collision_events = []
        for pair in pairs:
            key = pair[0].name.casefold()
            previous = winners.get(key)
            if previous is not None:
                collision_events.append((previous, pair))
            winners[key] = pair
        return list(winners.values()), collision_events

    @staticmethod
    def _wbc_full_name_map(pairs: List[Tuple[Path, Path]]):
        """Use stable source indices as namespaces so all raw WBC pairs remain addressable."""
        mapping = {}
        for index, (image, mask) in enumerate(pairs):
            image_name = f'{index:04d}__{image.name}'
            mask_name = f'{index:04d}__{image.stem}{mask.suffix}'
            mapping[image] = (image_name, mask_name)
        if len({names[0].casefold() for names in mapping.values()}) != len(pairs):
            raise RuntimeError("full-v2 failed to create unique WBC image names")
        if len({names[1].casefold() for names in mapping.values()}) != len(pairs):
            raise RuntimeError("full-v2 failed to create unique WBC mask names")
        return mapping

    @staticmethod
    def _relative_wbc_source(path: Path, wbc_root: Path) -> str:
        return path.relative_to(wbc_root).as_posix()

    def _build_wbc_manifest(self, protocol_id: str, wbc_root: Path,
                            prepared_by_split: dict, collision_events: dict):
        samples = []
        for split in ('train', 'val', 'test'):
            for image, mask, image_name, mask_name in prepared_by_split[split]:
                image_sha256 = self._sha256(image)
                mask_sha256 = self._sha256(mask)
                pair_sha256 = hashlib.sha256(
                    f'{image_sha256}:{mask_sha256}'.encode('ascii')
                ).hexdigest()
                samples.append({
                    'split': split,
                    'output_image': image_name,
                    'output_mask': mask_name,
                    'source_image': self._relative_wbc_source(image, wbc_root),
                    'source_mask': self._relative_wbc_source(mask, wbc_root),
                    'image_sha256': image_sha256,
                    'mask_sha256': mask_sha256,
                    'pair_sha256': pair_sha256,
                })

        overlaps = {}
        for field in ('source_image', 'image_sha256', 'mask_sha256', 'pair_sha256'):
            first_split = {}
            found = []
            for sample in samples:
                value = sample[field]
                earlier = first_split.setdefault(value, sample['split'])
                if earlier != sample['split']:
                    found.append({'value': value, 'splits': sorted({earlier, sample['split']})})
            overlaps[field] = found
        # An identical mask alone can be legitimate, but repeated source/image/pair content crosses
        # the sample-disjoint boundary and must fail closed.
        leakage_fields = ('source_image', 'image_sha256', 'pair_sha256')
        leaking = {field: overlaps[field] for field in leakage_fields if overlaps[field]}
        if leaking:
            raise RuntimeError(f'WBC cross-split leakage detected: {leaking}')

        collision_rows = {}
        for split, events in collision_events.items():
            collision_rows[split] = [
                {
                    'excluded_source_image': self._relative_wbc_source(old[0], wbc_root),
                    'kept_source_image': self._relative_wbc_source(new[0], wbc_root),
                    'output_image': new[0].name,
                }
                for old, new in events
            ]
        canonical_samples = json.dumps(
            samples, sort_keys=True, separators=(',', ':'), ensure_ascii=True
        ).encode('utf-8')
        return {
            'schema_version': 1,
            'dataset': 'wbc',
            'protocol_id': protocol_id,
            'split_seed': 42,
            'sample_manifest_sha256': hashlib.sha256(canonical_samples).hexdigest(),
            'sample_disjoint': True,
            'cross_split_overlap_counts': {
                field: len(values) for field, values in overlaps.items()
            },
            'collision_events': collision_rows,
            'samples': samples,
        }

    @staticmethod
    def _install_wbc_splits(dataset_dir: Path, staging_dir: Path):
        """Swap only generated image/mask directories, restoring the old set on failure."""
        backup_root = Path(tempfile.mkdtemp(prefix='.wbc_backup_', dir=dataset_dir))
        states = []
        try:
            for split in ('train', 'val', 'test'):
                for kind in ('images', 'masks'):
                    source = staging_dir / split / kind
                    target = dataset_dir / split / kind
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup = backup_root / split / kind
                    state = {'target': target, 'backup': None, 'installed': False}
                    if target.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        target.rename(backup)
                        state['backup'] = backup
                    states.append(state)
                    source.rename(target)
                    state['installed'] = True
        except Exception:
            for state in reversed(states):
                target, backup = state['target'], state['backup']
                if state['installed'] and target.exists():
                    shutil.rmtree(target)
                if backup is not None and backup.exists():
                    backup.rename(target)
            if backup_root.exists():
                shutil.rmtree(backup_root)
            raise
        else:
            shutil.rmtree(backup_root)

    # WBC dataset preparation with an explicit paper protocol and a separate full-data protocol.
    def _prepare_wbc(self, dataset_dir: Path, protocol_id: str) -> bool:
        raw_dir = dataset_dir / 'raw'
        wbc_dirs = sorted(p for p in raw_dir.glob('*WBC*') if p.is_dir())
        if not wbc_dirs:
            logging.error("WBC dataset directory not found")
            return False
        wbc_root = wbc_dirs[0]
        pairs = []
        for ds_dir in sorted(p for p in wbc_root.glob('Dataset*') if p.is_dir()):
            for bmp_path in sorted(ds_dir.glob('*.bmp')):    # look for the bmp images (sorting keeps the split reproducible)
                png_path = ds_dir / f"{bmp_path.stem}.png"  # the matching mask png
                if png_path.exists():
                    pairs.append((bmp_path, png_path))
        if not pairs:
            logging.error("No paired data found")
            return False
        tr, va, te = self._grouped_three_way_split(
            pairs, val_ratio=0.15, test_ratio=0.15, group_by=None, seed=42)
        nominal = {'train': tr, 'val': va, 'test': te}
        collision_events = {split: [] for split in nominal}
        if protocol_id == 'paper-legacy-v1':
            selected = {}
            for split, split_pairs in nominal.items():
                selected[split], collision_events[split] = \
                    self._resolve_wbc_legacy_collisions(split_pairs)
            output_names = None
        else:
            selected = nominal
            full_names = self._wbc_full_name_map(pairs)
            output_names = lambda image, mask: full_names[image]

        actual_counts = {split: len(split_pairs) for split, split_pairs in selected.items()}
        if len(pairs) == 400:
            expected = ({'train': 231, 'val': 58, 'test': 60}
                        if protocol_id == 'paper-legacy-v1'
                        else {'train': 280, 'val': 60, 'test': 60})
            if actual_counts != expected:
                raise RuntimeError(
                    f'{protocol_id} canonical WBC counts changed: {actual_counts} != {expected}'
                )

        with tempfile.TemporaryDirectory(prefix='.wbc_prepare_', dir=dataset_dir) as tmp:
            staging = Path(tmp)
            prepared_by_split = {}
            for split in ('train', 'val', 'test'):
                prepared_by_split[split] = self._copy_pairs_to_split(
                    selected[split], staging / split, output_names=output_names)
            manifest = self._build_wbc_manifest(
                protocol_id, wbc_root, prepared_by_split, collision_events)
            self._install_wbc_splits(dataset_dir, staging)

        installed_counts = {
            split: len([p for p in (dataset_dir / split / 'images').iterdir() if p.is_file()])
            for split in ('train', 'val', 'test')
        }
        if installed_counts != actual_counts:
            raise RuntimeError(f'installed WBC counts do not match the prepared set: {installed_counts}')
        manifest_path = dataset_dir / 'wbc_split_manifest.json'
        self._save_json(manifest_path, manifest)
        excluded_count = sum(len(events) for events in collision_events.values())
        self._save_dataset_info(dataset_dir, {
            'name': 'wbc',
            'num_classes': 3,
            'split': 'deterministic_per_image_3way',
            'split_seed': 42,
            'protocol_id': protocol_id,
            'protocol_description': (
                'Paper-release sample identities with explicit last-write-wins collision resolution.'
                if protocol_id == 'paper-legacy-v1'
                else 'Collision-safe full-data split for future experiments; not linked to paper metrics.'
            ),
            'raw_pair_count': len(pairs),
            'prepared_pair_count': sum(installed_counts.values()),
            'excluded_collision_pair_count': excluded_count,
            'nominal_split_counts': {split: len(items) for split, items in nominal.items()},
            'train_samples': installed_counts['train'],
            'val_samples': installed_counts['val'],
            'test_samples': installed_counts['test'],
            'sample_disjoint': manifest['sample_disjoint'],
            'cross_split_overlap_counts': manifest['cross_split_overlap_counts'],
            'manifest': manifest_path.name,
            'sample_manifest_sha256': manifest['sample_manifest_sha256'],
        })
        logging.info(
            "WBC protocol %s prepared: train %d / val %d / test %d (%d raw pairs, %d excluded collisions)",
            protocol_id, installed_counts['train'], installed_counts['val'],
            installed_counts['test'], len(pairs), excluded_count)
        return True

    # US_NERVE dataset preparation: generate empty masks (when there are none), then build the train/val/test splits
    def _prepare_us_nerve(self, dataset_dir: Path) -> bool:
        raw_dir = dataset_dir / 'raw'
        from PIL import Image
        import numpy as np
        img_files = list(raw_dir.rglob('*.tif')) + list(raw_dir.rglob('*.png')) + list(raw_dir.rglob('*.jpg'))
        pairs = []
        for img_path in img_files:
            mask_path = img_path.parent / (img_path.stem + '_mask' + img_path.suffix)
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        if not pairs:
            logging.warning("No masks found, generating empty masks")
            for img_path in img_files:
                img = Image.open(img_path)
                mask = np.zeros((img.height, img.width), dtype=np.uint8)  # all-black mask
                mpath = raw_dir / (img_path.stem + '_mask.png')
                Image.fromarray(mask).save(mpath)  # save the empty mask
                pairs.append((img_path, mpath))
        tr, va, te = self._grouped_three_way_split(pairs, val_ratio=0.1, test_ratio=0.1, group_by=None)
        self._copy_pairs_to_split(tr, dataset_dir / 'train')
        self._copy_pairs_to_split(va, dataset_dir / 'val')
        self._copy_pairs_to_split(te, dataset_dir / 'test')
        self._save_dataset_info(dataset_dir, {
            'name': 'us_nerve', 'num_classes': 1, 'split': 'per_image_3way',
            'train_samples': len(tr), 'val_samples': len(va), 'test_samples': len(te)
        })
        return True

    # VOC dataset preparation: read the official train/val file lists and copy the matching images and masks
    def _prepare_voc(self, dataset_dir: Path) -> bool:
        raw_dir = dataset_dir / 'raw'
        candidates = [
            raw_dir / 'VOCdevkit' / 'VOC2012',
            raw_dir / 'VOC2012'
        ]
        voc_root = None
        for c in candidates:
            if (c / 'JPEGImages').exists() and (c / 'SegmentationClass').exists() and (c / 'ImageSets' / 'Segmentation').exists():
                voc_root = c
                break
        if voc_root is None:
            logging.error("VOC2012 directory not found (JPEGImages, SegmentationClass and ImageSets/Segmentation are required)")
            return False

        jpeg_dir = voc_root / 'JPEGImages'
        seg_dir  = voc_root / 'SegmentationClass'
        set_dir  = voc_root / 'ImageSets' / 'Segmentation'

        def read_list(txt_path: Path):
            with open(txt_path, 'r') as f:
                return [x.strip() for x in f.readlines() if x.strip()]

        train_ids = read_list(set_dir / 'train.txt')
        val_ids   = read_list(set_dir / 'val.txt')

        def id_to_paths(_id):
            img = jpeg_dir / f'{_id}.jpg'
            msk = seg_dir  / f'{_id}.png'
            return img, msk

        train_pairs, val_pairs = [], []
        for _id in train_ids:
            img, msk = id_to_paths(_id)
            if img.exists() and msk.exists():
                train_pairs.append((img, msk))
        for _id in val_ids:
            img, msk = id_to_paths(_id)
            if img.exists() and msk.exists():
                val_pairs.append((img, msk))

        if len(train_pairs) == 0 or len(val_pairs) == 0:
            logging.error("The VOC train/val lists are empty, or no paired images/masks were found")
            return False

        self._copy_pairs_to_split(train_pairs, dataset_dir / 'train')
        self._copy_pairs_to_split(val_pairs,   dataset_dir / 'val')

        self._save_dataset_info(dataset_dir, {
            'name': 'voc', 'num_classes': 21,
            'train_samples': len(train_pairs),
            'val_samples': len(val_pairs),
            'description': 'Pascal VOC2012 segmentation (JPEGImages + SegmentationClass)'
        })
        logging.info("VOC dataset prepared!")
        return True

    # Generic dataset preparation: match images and masks as the config says, then build the train/val/test splits
    def _prepare_generic(self, dataset_dir: Path, config: dict) -> bool:
        raw_dir = dataset_dir / 'raw'
        img_pat = config.get('image_pattern', '*.jpg')
        msk_pat = config.get('mask_pattern', '*.png')
        # Top level first, recurse only if empty. Kaggle's carvana train.zip carries its own `train/` level inside (all 5089/5089
        # members sit under it) and _auto_extract_raw keeps that structure as is ⇒ a non-recursive glob alone matches 0 images at
        # the top level and then fails silently (this function used to exit 0; see main()). WBC/us_nerve already recurse, this is the exception.
        # ⚠️ rglob must not be unconditional: the top level of raw/ here also holds a hand-flattened copy, and the same tree hides
        # Carvana's 100k **unlabeled test images** — recursing unconditionally hauls those in too (measured 5088 → 105,152),
        # which both bogs down the pairing loop and reorders images ⇒ the grouped split changes ⇒ every existing checkpoint mismatches.
        # Two exclusive paths: images up top take the old route (bit-identical here), an empty top level recurses (the stranger's case).
        images = sorted(raw_dir.glob(img_pat))
        if not images:
            images = sorted(raw_dir.rglob(img_pat))
            if images:
                logging.info("no %s at the top level of raw/; found %d in subdirectories (the archive kept its internal layout)",
                             img_pat, len(images))
        # Index the masks in one pass: an rglob per image is O(n²) and becomes unusable on a tree of 100k files.
        mask_index = {}
        for m in sorted(raw_dir.rglob(msk_pat)):
            mask_index.setdefault(m.name, m)       # on a name clash keep the first in traversal order, for determinism
        pairs = []
        for img_path in images:
            m = mask_index.get(msk_pat.replace('*', img_path.stem))
            if m:
                pairs.append((img_path, m))
        if not pairs:
            logging.error(
                "No paired data found: nothing under %s (subdirectories included) matched an image-mask pair for %s / %s. "
                "Check that both archives have been placed in that directory (their internal structure is detected automatically).",
                raw_dir, img_pat, msk_pat)
            return False
        tr, va, te = self._grouped_three_way_split(
            pairs, val_ratio=config.get('val_ratio', 0.1),
            test_ratio=config.get('test_ratio', 0.1), group_by=config.get('group_by'))
        self._copy_pairs_to_split(tr, dataset_dir / 'train')
        self._copy_pairs_to_split(va, dataset_dir / 'val')
        self._copy_pairs_to_split(te, dataset_dir / 'test')
        self._save_dataset_info(dataset_dir, {
            'name': dataset_dir.name, 'split': f"group_by={config.get('group_by')}_3way",
            'train_samples': len(tr), 'val_samples': len(va), 'test_samples': len(te)
        })
        return True

    # Prepare every dataset in the config
    def prepare_all(self, force: bool = False, wbc_protocol: str = 'paper-legacy-v1'):
        ok_all = True
        for name in self.dataset_configs:
            logging.info(f"\n{'=' * 60}\nPreparing the {name.upper()} dataset\n{'=' * 60}")
            ok_all &= bool(self.prepare_dataset(name, force, wbc_protocol=wbc_protocol))
        return ok_all

def main():
    parser = argparse.ArgumentParser(description="Semantic segmentation dataset preparation tool")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'carvana', 'wbc', 'us_nerve', 'voc'])  # the supported dataset options
    parser.add_argument('--data_root', type=str, default='./data')  # data root directory
    parser.add_argument('--force', action='store_true')  # whether to force the dataset to be prepared again
    parser.add_argument(
        '--wbc_protocol', choices=SegmentationDatasetPreparer.WBC_PROTOCOLS,
        default='paper-legacy-v1',
        help=('paper-legacy-v1 reproduces the released 231/58/60 WBC sample identities; '
              'full-v2 keeps all 400 raw pairs with collision-safe filenames for new experiments'),
    )
    args = parser.parse_args()

    preparer = SegmentationDatasetPreparer(args.data_root)
    if args.dataset == 'all':
        ok = preparer.prepare_all(args.force, args.wbc_protocol)  # prepare every dataset
    else:
        ok = preparer.prepare_dataset(
            args.dataset, args.force, wbc_protocol=args.wbc_protocol)  # prepare the requested dataset
    # Respect the return value: main() used to throw it away, so "No paired data found" printed a single error line and then
    # exited with code 0, handing anyone following the README an empty dataset and a "success". Failure has to be loud.
    if not ok:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
