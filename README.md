# EAGLE
Repository in Support of EAGLE Submission


## Setting Up the Environment

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/chadvanderbilt/EAGLE.git
   ```

2. **Create and Activate a Conda Environment:**
   ```bash
   conda create -n irt_env python=3.9 --file requirements.txt
   conda activate irt_env
   ```




## Training (Fine-Tuning)
We fine-tuned the tile-level encoder of [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) and used Gated MIL attention ([Ilse et al. 2019](https://arxiv.org/abs/1802.04712)) as slide-level aggregator to predict *EGFR* mutational status in lung adenocarcinoma. The fine-tuning algorithm is described in detail in [Campanella et al. 2024](https://arxiv.org/abs/2403.04865). Training was parallelized over 24 H100 GPUs. We used the LSF scheduler for job submission. The training submission script that we used can be found at `training/train.lsf`. Note: the script was redacted to remove cluster-specific details. LSF parameters may need to be modified. The training can be performed by
```bash
cd training
bsub < train.lsf
```
with lsf or for slurm hpc:

```bash
cd training
sbatch train.slurm
```
To run correctly, the pipeline expects the path of two files to be included in the `training/datasets.py` script:
- `slide_data.csv`: a file containing the name, path, target, and data split for each slide in the dataset.
- `tile_data.csv`: a file containing the slide name, x, and y coordinates as well as the level (within the slide file) and a rescale factor for each tile in the dataset. The rescale factor may be necessary if the right magnification is not available in the slide.

To finetune GigaPath, access to their weights is necessary via HuggingFace. Package requirements in conda irt_env.


## Model Calibration

This section explains how to select the model thresholds for deployment in a new institution. We provide an example csv file: `calibration/test_data.csv`.
The script `calibration/pretrial_tuning.py` illustrates the analysis we performed. To simulate the deployment of EAGLE one can do:
```python
import numpy as np
import pandas as pd
import utils
df = pd.read_csv('example_data.csv')
utils.simulation(df)
```

To evaluate the performance of the assisted workflow after selecting the thresholds one can do:
```python
import numpy as np
import pandas as pd
import utils
df = pd.read_csv('example_data.csv')
threshold_npv = 0.023
threshold_ppv = 0.997
utils.get_performance_assisted_bootstrapped(df, th0=[threshold_npv], th1=[threshold_ppv], n=1000, target_col='target', rapid_col='rapid', eagle_col='score')
```

More details about parameters of these functions can be found by:
```python
import utils
help(utils.simulation)
help(utils.plot_simulation)
help(utils.get_performance_assisted_bootstrapped)
```

## Running the IRT Pipeline

This section explains how to set up and run the IRT Pipeline for monitoring and processing scanned slides in real-time. The repository is hosted at [EAGLE GitHub Repository](https://github.com/chadvanderbilt/EAGLE.git) and the relevant shell scripts are located in the `IRT_Pipeline` subfolder.

### 1. **Monitoring Script (`run.sh`)**

The `run.sh` script monitors newly scanned slides every hour and generates manifests for downstream processing.

**Cron Job Setup:**

To schedule this script to run every hour, offset by 30 minutes from the `run_full_gigapath_pipeline.sh`, add the following to your crontab:

```bash
crontab -e
```

Add this line to schedule `run.sh`:
```bash
30 * * * * /path/to/EAGLE/IRT_Pipeline/run.sh
```

### 2. **Full Pipeline Script (`run_full_gigapath_pipeline.sh`)**

The `run_full_gigapath_pipeline.sh` script processes all slides identified by the monitoring script.

**Cron Job Setup:**

To run the full pipeline every hour on the hour, add the following to your crontab:

```bash
0 * * * * /path/to/EAGLE/IRT_Pipeline/run_full_gigapath_pipeline.sh
```

### Directory Structure

The scripts assume the following directory structure for outputs and logs:

```
/your/production/
├── logs/
│   └── irt_monitor/         # Logs generated by the monitoring script
├── slide_data/              # Manifests and slide data outputs
└── run_EGFR/                # Directory where the pipeline outputs. BASE_DIR is run_EGFR
    ├── molecular_watcher/   # Directory for monitoring molecular data
    ├── slides_to_run/       # Directory containing slides to be processed
    ├── EGFR_results/        # Directory for EGFR test results
    ├── tmp_files/           # Temporary files directory
    └── checkpoints/         # Model checkpoints directory
```


Ensure these directories exist or are correctly specified in the shell scripts. Adjust paths as needed for your environment. 
Our laboratory has API endpoints available for Real Time data.  If such data is available via database then adjust as needed. 

### Logging

Each execution of the `run.sh` script will generate a log file in `/your/production/logs/irt_monitor/`, with filenames formatted as `YYYY-MM-DD_HH-MM-SS.log`.

### Example Run

To manually execute the monitoring script and generate manifests:
```bash
bash /path/to/EAGLE/IRT_Pipeline/run.sh
```

To manually execute the full pipeline script:
```bash
bash /path/to/EAGLE/IRT_Pipeline/run_full_gigapath_pipeline.sh
```
