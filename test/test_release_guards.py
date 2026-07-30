import copy
import io
import json
from pathlib import Path
import stat
import tarfile
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat
from scipy.linalg import hadamard

from scripts.merge_results import clean_eval_row
from scripts.predict_from_bucket import load_bucket
from scripts.predict_reconstruct import load_buckets
from scripts.prepare_seg_datasets import SegmentationDatasetPreparer
from scripts.recon_dump import (
    generation_signature,
    to_uint8_gray,
    validate_existing_generation,
)
from scripts.reconstruct_eval import validate_checkpoint_config
from scripts.validate_run_artifacts import write_manifest
from src.reconstruction.gi import trad_gi_recon
from src.datasets.folder_dataset import FolderSegDataset
from src.trainer import SegmentationTrainer
from src.utils import checkpoint as checkpoint_utils
from src.utils.ghost_patterns import get_hadamard_matrix
from src.utils.config_parser import load_config
from src.utils.reconstruction_manifest import (
    build_reconstruction_inventory,
    inventory_sha256,
)


def test_checkpoint_loader_does_not_implicitly_fallback_to_pickle(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs['weights_only'])
        if kwargs['weights_only']:
            raise ValueError('legacy object')
        return {'legacy': True}

    monkeypatch.setattr(checkpoint_utils.torch, 'load', fake_load)
    with pytest.raises(RuntimeError, match='allow_unsafe_pickle'):
        checkpoint_utils.load_checkpoint('legacy.pth')
    assert calls == [True]

    with pytest.warns(RuntimeWarning, match='arbitrary code'):
        loaded = checkpoint_utils.load_checkpoint(
            'trusted-legacy.pth', allow_unsafe_pickle=True)
    assert loaded == {'legacy': True}
    assert calls == [True, True, False]


def test_mat_bucket_ambiguity_requires_explicit_key(tmp_path):
    path = tmp_path / 'capture.mat'
    savemat(path, {
        'B': np.arange(8, dtype=np.float32),
        'd_B': np.arange(8, dtype=np.float32)[::-1],
    })
    with pytest.raises(ValueError, match='Ambiguous MAT file'):
        load_bucket(str(path), 8)

    selected = load_bucket(str(path), 8, mat_key='d_B')
    assert selected[0] == pytest.approx(1.0)
    assert selected[-1] == pytest.approx(0.0)


@pytest.mark.parametrize('suffix', ['.mat', '.npz'])
def test_reconstruction_bucket_archives_fail_closed_on_multiple_keys(tmp_path, suffix):
    path = tmp_path / f'capture{suffix}'
    arrays = {
        'B': np.arange(8, dtype=np.float32),
        'd_B': np.arange(8, dtype=np.float32)[::-1],
    }
    if suffix == '.mat':
        savemat(path, arrays)
    else:
        np.savez(path, **arrays)

    with pytest.raises(ValueError, match='Ambiguous'):
        load_buckets(path)
    selected = load_buckets(path, mat_key='d_B')
    assert selected.shape == (1, 8)
    np.testing.assert_array_equal(selected[0], arrays['d_B'])


def _minimal_config():
    return {
        'model': {'name': 'BaselineUNetPP', 'params': {'classes': 1}},
        'data': {
            'dataset': 'carvana', 'img_size': 128, 'classes': 1,
            'bucket_size': 512,
        },
        'inference': {'threshold': 0.5},
    }


def test_reconstruct_eval_rejects_load_bearing_config_mismatch():
    external = _minimal_config()
    validate_checkpoint_config(external, copy.deepcopy(external))

    embedded = copy.deepcopy(external)
    embedded['data']['bucket_size'] = 256
    with pytest.raises(ValueError, match='data.bucket_size'):
        validate_checkpoint_config(external, embedded)


@pytest.mark.parametrize('N,M', [(8, 3), (8, 8), (16, 5), (32, 9)])
def test_hadamard_rectangle_matches_scipy_prefix(N, M):
    order = 1 << (max(N, M) - 1).bit_length()
    expected = hadamard(order, dtype=np.int8)[:M, :N].astype(np.float32)
    actual = get_hadamard_matrix(N, M)
    assert actual.shape == (M, N)
    np.testing.assert_array_equal(actual, expected)


def test_segmentation_archive_extraction_rejects_links(tmp_path):
    preparer = SegmentationDatasetPreparer.__new__(SegmentationDatasetPreparer)
    zip_path = tmp_path / 'linked.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        link = zipfile.ZipInfo('link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, '../outside')
    with pytest.raises(ValueError, match='symbolic link'):
        preparer._extract_archive(zip_path, tmp_path / 'zip_out')

    tar_path = tmp_path / 'linked.tar'
    with tarfile.open(tar_path, 'w') as archive:
        regular = tarfile.TarInfo('inside.txt')
        regular.size = 2
        archive.addfile(regular, io.BytesIO(b'ok'))
        link = tarfile.TarInfo('link')
        link.type = tarfile.SYMTYPE
        link.linkname = '../outside'
        archive.addfile(link)
    with pytest.raises(ValueError, match='non-regular member'):
        preparer._extract_archive(tar_path, tmp_path / 'tar_out')


def test_gi_direct_acquired_rows_matches_old_zero_padded_reference():
    img_size, M, batch = 4, 5, 3
    N = img_size * img_size
    rng = np.random.default_rng(4)
    buckets = rng.normal(size=(batch, M)).astype(np.float32)
    patterns = get_hadamard_matrix(N, M)

    old_buckets = np.zeros((batch, N), dtype=np.float32)
    old_buckets[:, :M] = buckets
    reference = (old_buckets @ hadamard(N).astype(np.float32)).reshape(
        batch, img_size, img_size)
    reference -= reference.mean(axis=(1, 2), keepdims=True)
    reference = ((reference - reference.min(axis=(1, 2), keepdims=True)) /
                 (reference.max(axis=(1, 2), keepdims=True)
                  - reference.min(axis=(1, 2), keepdims=True) + 1e-8))

    actual = trad_gi_recon(patterns, buckets, img_size, device='cpu')[:, 0]
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-6)


def test_recon_dump_generation_manifest_fails_closed(tmp_path):
    meta_path = tmp_path / '_dump_meta.json'
    clean = {'method': 'hsi', 'noise': {'mode': 'clean'}}
    noisy = {'method': 'hsi', 'noise': {'mode': 'fixed_db', 'snr_db': 20}}
    clean_signature = generation_signature(clean)
    assert clean_signature != generation_signature(noisy)

    with pytest.raises(RuntimeError, match='no generation manifest'):
        validate_existing_generation(meta_path, True, clean_signature)

    meta_path.write_text(json.dumps({'generation_signature': 'different'}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='does not match'):
        validate_existing_generation(meta_path, True, clean_signature)

    meta_path.write_text(
        json.dumps({'generation_signature': clean_signature}), encoding='utf-8')
    assert validate_existing_generation(meta_path, True, clean_signature)


def test_recon_dump_refuses_nonfinite_pixels():
    with pytest.raises(FloatingPointError, match='NaN or Inf'):
        to_uint8_gray(np.array([[np.nan]], dtype=np.float32))


def test_reconstruction_dataset_requires_complete_hash_inventory(tmp_path):
    root = tmp_path / 'data_recon_case' / 'hsi_mnist'
    split = root / 'train'
    (split / 'images').mkdir(parents=True)
    (split / 'masks').mkdir()
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(
        split / 'images' / 'sample.png')
    Image.fromarray(np.array([[0, 255], [0, 255]], dtype=np.uint8)).save(
        split / 'masks' / 'sample.png')
    meta = split / '_dump_meta.json'
    meta.write_text(json.dumps({
        'generation_signature': 'a' * 64,
        'complete': False,
        'status': 'partial_limit',
    }), encoding='utf-8')
    with pytest.raises(RuntimeError, match='not complete'):
        FolderSegDataset(
            root, bucket_size=1, img_size=2, num_classes=1,
            mode='train', compute_bucket=False, num_workers=0,
        )

    inventory = build_reconstruction_inventory(split)
    meta.write_text(json.dumps({
        'generation_signature': 'a' * 64,
        'complete': True,
        'status': 'complete',
        'total_in_split': 1,
        'processed': 1,
        'pair_count': 1,
        'file_count': 2,
        'inventory_sha256': inventory_sha256(inventory),
        'inventory': inventory,
    }), encoding='utf-8')
    dataset = FolderSegDataset(
        root, bucket_size=1, img_size=2, num_classes=1,
        mode='train', compute_bucket=False, num_workers=0,
    )
    assert len(dataset) == 1

    Image.fromarray(np.full((2, 2), 127, dtype=np.uint8)).save(
        split / 'images' / 'sample.png')
    with pytest.raises(RuntimeError, match='inventory or SHA-256 mismatch'):
        FolderSegDataset(
            root, bucket_size=1, img_size=2, num_classes=1,
            mode='train', compute_bucket=False, num_workers=0,
        )


def test_explicit_reconstruction_contract_survives_custom_root_rename(tmp_path):
    root = tmp_path / 'renamed_custom'
    split = root / 'train'
    (split / 'images').mkdir(parents=True)
    (split / 'masks').mkdir()
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(split / 'images' / 'a.png')
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(split / 'masks' / 'a.png')
    with pytest.raises(RuntimeError, match='cannot read reconstruction manifest'):
        FolderSegDataset(
            root, bucket_size=1, img_size=2, num_classes=1, mode='train',
            compute_bucket=False, num_workers=0,
            require_reconstruction_manifest=True,
        )


def test_every_released_ta_config_explicitly_requires_reconstruction_manifest():
    config_dir = Path(__file__).parents[1] / 'configs' / 'experiments'
    ta_configs = sorted(config_dir.glob('ta_*.yaml'))
    assert len(ta_configs) == 38
    assert all(
        load_config(path)['data'].get('require_reconstruction_manifest') is True
        for path in ta_configs
    )


def test_tpls_adaptive_schedule_boundaries():
    trainer = SegmentationTrainer.__new__(SegmentationTrainer)
    trainer.use_adaptive = True
    trainer.base_seg_weight = 2.0
    trainer.base_aux_weight = 3.0
    trainer.stage1_epochs = 2
    trainer.stage2_epochs = 3
    trainer.stage1_weights = (0.1, 0.9)
    trainer.stage2_weights = (0.5, 0.5)
    trainer.stage3_weights = (0.9, 0.1)

    assert trainer._get_adaptive_weights(1) == pytest.approx((0.2, 2.7))
    assert trainer._get_adaptive_weights(2) == pytest.approx((0.2, 2.7))
    assert trainer._get_adaptive_weights(3) == pytest.approx((1.0, 1.5))
    assert trainer._get_adaptive_weights(5) == pytest.approx((1.0, 1.5))
    assert trainer._get_adaptive_weights(6) == pytest.approx((1.8, 0.3))


def test_trainer_rejects_nonfinite_loss_before_optimizer_step(tmp_path):
    class BadModel(torch.nn.Module):
        input_type = 'image'

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            return {'logits': value * self.weight, 'aux_recon': None}

    model = BadModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        'image': torch.ones(1, 1, 2, 2),
        'mask': torch.ones(1, 2, 2, dtype=torch.long),
    }
    config = {
        'data': {'classes': 1, 'bucket_on_gpu': False, 'bucket_noise_snr_db': None},
        'training': {
            'epochs': 1, 'amp': False, 'gradient_clip': 0.0,
            'checkpoint_dir': str(tmp_path / 'checkpoints'),
            'experiment_name': 'bad',
            'loss': {
                'seg_weight': 1.0, 'aux_recon_weight': 0.0,
                'adaptive_loss': {'enable': False},
            },
        },
        'logging': {'use_tensorboard': False},
        'inference': {'threshold': 0.5},
        'model': {'name': 'BadModel'},
    }
    criterion = lambda logits, mask: logits.mean() * torch.tensor(float('nan'))
    trainer = SegmentationTrainer(
        model, optimizer, criterion, criterion, [batch], [batch], None,
        torch.device('cpu'), config,
    )
    with pytest.raises(FloatingPointError, match='non-finite training loss'):
        trainer._train_epoch(1, 1.0, 0.0)


def _car_ids(items):
    return {image.stem.rsplit('_', 1)[0] for image, _ in items}


def test_grouped_split_is_deterministic_and_car_disjoint():
    pairs = []
    for car in range(12):
        for view in range(2):
            image = Path(f'car{car:02d}_{view:02d}.jpg')
            pairs.append((image, Path(f'car{car:02d}_{view:02d}_mask.gif')))
    preparer = SegmentationDatasetPreparer.__new__(SegmentationDatasetPreparer)

    first = preparer._grouped_three_way_split(
        pairs, val_ratio=0.2, test_ratio=0.2, group_by='carid', seed=42)
    second = preparer._grouped_three_way_split(
        pairs, val_ratio=0.2, test_ratio=0.2, group_by='carid', seed=42)
    assert first == second
    train_ids, val_ids, test_ids = map(_car_ids, first)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert sum(map(len, first)) == len(pairs)


def test_merge_results_accepts_clean_eval_json():
    data = {
        'experiment_name': 'rev_carvana_tpls',
        'dataset': 'carvana',
        'bucket_size': 512,
        'train_seed': 42,
        'ckpt': 'checkpoints/rev_carvana_tpls_s42/best.pth',
        'noise_snr_db': None,
        'metrics': {'miou_fg': 0.75, 'mdice_fg': 0.86},
    }
    row = clean_eval_row(data, '_rev/results/rev_carvana_tpls_s42.json')
    assert row is not None
    assert row['experiment'] == 'rev_carvana_tpls_s42'
    assert row['family'] == 'main_clean'
    assert row['seed'] == '42'
    assert row['miou_fg'] == '75.0000'


def test_run_artifact_validator_checks_contract_and_hashes(tmp_path):
    artifact = tmp_path / 'results' / 'eval.json'
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({
        'split': 'test', 'dataset': 'mnist', 'train_seed': 42,
        'model': 'GRUUNetPP', 'experiment_name': 'rev_mnist_tpls_s42',
        'samples': 10, 'metrics': {
            'pa': 0.9, 'mpa': 0.8, 'miou': 0.7, 'mdice': 0.8,
            'miou_fg': 0.5, 'mdice_fg': 0.6,
        },
    }), encoding='utf-8')
    spec = 'results/eval.json|test|mnist|42|segmentation|GRUUNetPP|rev_mnist_tpls_s42'
    payload = write_manifest('smoke', [spec], Path('results/manifest.json'), root=tmp_path)
    assert payload['expected_artifact_count'] == 1
    assert len(payload['artifacts'][0]['sha256']) == 64
    assert (tmp_path / 'results' / 'manifest.json').exists()

    wrong = 'results/eval.json|test|mnist|43|segmentation|GRUUNetPP|rev_mnist_tpls_s42'
    with pytest.raises(ValueError, match='train_seed'):
        write_manifest('smoke', [wrong], Path('results/invalid.json'), root=tmp_path)

    wrong_model = (
        'results/eval.json|test|mnist|42|segmentation|FCNUNetPP|rev_mnist_tpls_s42'
    )
    with pytest.raises(ValueError, match='model'):
        write_manifest('smoke', [wrong_model], Path('results/invalid-model.json'), root=tmp_path)

    content = json.loads(artifact.read_text(encoding='utf-8'))
    content['metrics'] = {'miou_fg': 0.5}
    artifact.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(ValueError, match='metrics must contain exactly'):
        write_manifest('smoke', [spec], Path('results/truncated.json'), root=tmp_path)
