# EAGLE
Repository in Support of EAGLE Submission

## Training (Fine-Tuning)
We fine-tuned the tile-level encoder of [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) and used Gated MIL attention ([Ilse et al. 2019](https://arxiv.org/abs/1802.04712)) as slide-level aggregator to predict *EGFR* mutational status in lung adenocarcinoma. The fine-tuning algorithm is described in detail in [Campanella et al. 2024](https://arxiv.org/abs/2403.04865). Training was parallelized over 24 H100 GPUs. We used the LSF scheduler for job submission. The training submission script that we used can be found at `training/train.lsf`. Note: the script was redacted to remove cluster-specific details. LSF parameters may need to be modified. The training can be performed by
```bash
cd training
bsub < train.lsf
```
To run correctly, the pipeline expects the path of two files to be included in the `training/datasets.py` script:
- `slide_data.csv`: a file containing the name, path, target, and data split for each slide in the dataset.
- `tile_data.csv`: a file containing the slide name, x, and y coordinates as well as the level (within the slide file) and a rescale factor for each tile in the dataset. The rescale factor may be necessary if the right magnification is not available in the slide.

To finetune GigaPath, access to their weights is necessary via HuggingFace. Package requirements:
```
PIL
torch
torchvision
cucim
numpy
pandas
sklearn
timm
```
