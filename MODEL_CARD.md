# Released checkpoint card

The canonical, detailed model card is maintained with the weights:

- https://huggingface.co/hanyuyuan/STSF-TPLS-weights
- Audited checkpoint revision used by the download examples: `22a6e8ee71212ed4574b1a35a5c27e0681219dba`
- Code revision recorded by the weight card: `de7a83fc5b2ae3aa292603671d316fd604b521f4`

## License separation

- Source code in this GitHub repository: MIT, see [LICENSE](LICENSE).
- Released checkpoint weights on Hugging Face: CC BY 4.0, as declared in the canonical model card.
- Training and evaluation datasets: source-specific terms, see [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Intended use and limitations

The checkpoints are research artifacts for reproducing the experiments described in arXiv:2607.22077. They assume the documented natural-order Sylvester Hadamard acquisition, 128x128 scenes, and M = 512 measurements. They are not intended for clinical, diagnostic, safety-critical, or unvalidated out-of-domain deployment. WBC outputs are research segmentation results, not medical decisions.

Load only checkpoints obtained from the immutable revision and verified against `MANIFEST.json`. Keep `weights_only=True`; do not silently fall back to general pickle loading.

This pointer is intentionally short so intended-use details and checkpoint inventories do not drift between GitHub and Hugging Face.
