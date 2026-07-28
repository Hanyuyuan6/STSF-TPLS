import os, random, numpy as np, torch  # os, random, numpy and PyTorch


def seed_everything(seed=42):
    """Set every random source + turn on determinism, so runs are reproducible (reviewer request)."""
    os.environ['PYTHONHASHSEED'] = str(seed)  # fix the hash seed
    # cuBLAS deterministic workspace: when running `python -m scripts.train` directly without
    # exporting it in the shell first, this keeps GEMM from silently falling back to a non-deterministic implementation (equivalent to the export in the README, does not override a value the user already set).
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)  # Python built-in random
    np.random.seed(seed)  # numpy random
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # all GPUs
    # cuDNN determinism: disable benchmark kernel auto-selection, force deterministic convolution algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # global deterministic algorithms; warn_only=True so the odd operator without a deterministic implementation (e.g. the backward pass of bilinear upsampling) does not raise and abort training
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    """DataLoader worker init function; paired with a fixed-seed generator it makes multi-process shuffling/augmentation reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
