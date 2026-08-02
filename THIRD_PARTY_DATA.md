# Third-party datasets

The repository does not redistribute dataset files. Its MIT license applies to the repository's source code and author-owned documentation, not to Carvana, MNIST, WBC images, labels, or masks.

| Dataset | Acquisition source | Provenance and rights boundary |
|---|---|---|
| Carvana | https://www.kaggle.com/c/carvana-image-masking-challenge | Download through Kaggle. Competition access and reuse are governed by the rules and terms displayed to the downloader; do not assume the code license applies to the data. |
| MNIST | https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html | `torchvision.datasets.MNIST(..., download=True)` retrieves MNIST into the user's local data root. Torchvision's documentation identifies the dataset and download behavior but does not make this repository its licensor. |
| WBC | https://github.com/zxaoyou/segmentation_WBC | The upstream repository displays a GPL-3.0 license, but its README also attributes Dataset 1 images to Jiangxi Tecom Science Corporation and Dataset 2 images to the CellaVision blog. Preserve the requested acknowledgements and verify the upstream/rightsholder terms for the intended use. |

## Required scholarly attribution

For WBC, follow the upstream README's acknowledgements and cite:

> Zheng, X., Wang, Y., Wang, G. & Liu, J. Fast and robust segmentation of white blood cell images by self-supervised learning. *Micron* **107**, 55-71 (2018). https://doi.org/10.1016/j.micron.2018.01.010

For Carvana and MNIST, cite the dataset/source required by the venue and the terms shown by the data provider at download time.

## Local-only handling

- Place raw or prepared datasets under `data/` or `data_rev/`; both are ignored by Git.
- Do not commit subject data, local download credentials, Kaggle tokens, extracted archives, or generated split trees.
- Generated `dataset_info.json` and `wbc_split_manifest.json` are reproducibility records for local runs; review them for machine-local paths before publishing them elsewhere.
