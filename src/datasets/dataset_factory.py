from pathlib import Path  # Path, for cross-platform path handling
from .folder_dataset import FolderSegDataset  # the generic dataset class
from .specific_datasets import (  # the dataset-specific classes
    CarvanaDataset,
    WBCDataset,
    USNerveDataset,
    VOCDataset,
    MNISTSegDataset
)
import logging  # the logging module, for the log output

# dataset registry, keyed by name with the matching dataset class as the value
DATASET_REGISTRY = {
    'carvana': CarvanaDataset,  # Carvana car segmentation dataset
    'wbc': WBCDataset,          # white blood cell dataset
    'us_nerve': USNerveDataset, # ultrasound nerve dataset
    'voc': VOCDataset,          # Pascal VOC dataset
    'mnist': MNISTSegDataset,   # MNIST binary segmentation (torchvision downloads it automatically)
    'custom': FolderSegDataset, # generic folder-format dataset (custom datasets)
}

def get_dataset(dataset_name, **kwargs):
    # look the matching dataset class up in the registry
    DatasetClass = DATASET_REGISTRY.get(dataset_name)

    # if it is not there, warn and fall back to the generic folder class
    if DatasetClass is None:
        logging.warning(
            f"Unknown dataset '{dataset_name}', falling back to the generic format. "
            f"Supported datasets: {list(DATASET_REGISTRY.keys())}"
        )
        DatasetClass = FolderSegDataset

    # take the root_dir argument and turn it into a Path when it is there
    root_dir = kwargs.get('root_dir')
    if root_dir:
        kwargs['root_dir'] = Path(root_dir)

    try:
        # instantiate the dataset, forwarding every argument
        dataset = DatasetClass(**kwargs)
        # record the successful creation and the number of samples
        logging.info(
            f"Dataset created: {DatasetClass.__name__} "
            f"({len(dataset)} samples)"
        )
        # if the dataset has a validate method, call it to check the integrity and warn on failure
        if hasattr(dataset, 'validate'):
            if not dataset.validate():
                logging.warning("Dataset validation flagged problems, please check the data integrity")

        return dataset  # return the dataset instance

    except Exception as e:
        # on failure, log the error and re-raise
        logging.error(f"Failed to create dataset {dataset_name}: {e}")
        raise

def register_dataset(name, dataset_class):
    # if the name is already taken, warn that it is being overridden
    if name in DATASET_REGISTRY:
        logging.warning(f"Overriding an existing dataset: {name}")

    # register or override the dataset class
    DATASET_REGISTRY[name] = dataset_class
    # record the registration
    logging.info(f"Registered dataset: {name} -> {dataset_class.__name__}")

def list_datasets():
    # return the list of every registered dataset name
    return list(DATASET_REGISTRY.keys())