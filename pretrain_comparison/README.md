# pretrain_comparison


# Create the environment using a custom name
```bash
conda create SleepBench python=3.10
conda activate SleepBench
cd pretrain_comparison
pip install -r requirements.txt
```

## 📥 Data Download

Please download the data from: `<LINK>`

Then, run the following to update the data split:

```bash
python /pretrain_comparison/comparison/config/update_data_split.py --data_path </your/data/path>
```

## initlize the config files

please run:
cd /pretrain_comparison
```bash
python init.py --data_path </your/data/path>
```

## 🔁 Pre-training

Pretraining should be done on **two NVIDIA A100 80 GB GPUs** for optimal compatibility. If using fewer or smaller GPUs, adjust the batch size accordingly.

Navigate to `pretrain_comparison/comparison/pipeline` and run one of the following based on your model:
#### example run
torchrun --nproc_per_node=1 --master_port=29501 main_cl.py --config /oak/stanford/groups/mignot/projects/SleepBenchTest/pretrain_comparison/comparison/config/config_CL.yaml

### 🔹 Contrastive Learning (CL)
- `main_cl.py`
> Toggle between `CL LOO` and `CL Pairwise` in `pretrain_comparison/comparison/config/config_CL.yaml` under the `model` field.


### 🔹 Masked Autoencoders (MAE)
- `main_MAE.py` – MAE (Time, all patches) use config: `pretrain_comparison/comparison/config/config_multimodalMAE.yaml`
- `main_masked.py` – MAE (Time, masked patches) use config: `pretrain_comparison/comparison/config/config_multimodalMAE_masked.yaml`
- `main_fft.py` – MAE (Freq, all patches) use config: `pretrain_comparison/comparison/config/config_fft_MAE.yaml`
- `main_fft_masked.py` – MAE (Freq, masked patches) use config: `pretrain_comparison/comparison/config/config_fft_MAE_masked.yaml`

### 🔹 Denoising Autoencoders (DAE)
- `main_noise.py` – DAE (Time) use config: `pretrain_comparison/comparison/config/config_noise.yaml`
- `main_fft_noise.py` – DAE (Freq) use config: `pretrain_comparison/comparison/config/config_fft_noise.yaml`

---


## ✨ Generate Embeddings


To generate embeddings, navigate to `pretrain_comparison/comparison/pipeline` and select models using the `models_list` in the inital lines of the script then run:

#### example run
python gen_embeddings_cl.py

- `gen_embeddings_cl.py` – for CL subtypes
- `gen_embeddings.py`  – for MAE and DAE subtypes


## 🧪 Fine-Tuning

Fine-tuning is optimized for a **single NVIDIA A100 80 GB GPU**. Adjust batch size if using a different setup.

Update set the folder name of the `pretrain_type` in the `/pretrain_comparison/fine_tune/config_fine_tune.yaml`. You can find folders of the models you have made embeddings for in `pretrain_comparison/output/final_embeddings` folder. Note that there should only be a single run for any pretraining type (i.e. only one folder `pretrain_comparison/output/final_embeddings/<pretrain_type>/`).

Then, open and run the appropriate notebook:

- `fine_tune/ahi_diagnosis/fine_tune_ahi.ipynb`
- `fine_tune/sleep_stage_and_age/fine_tune_sleep_stage_and_age.ipynb`
- `fine_tune/death_and_diagnosis/fine_tune_diagnosis.ipynb`

---


## 📊 Evaluate Performance

to evaluate the performance please cd into `/pretrain_comparison/evaluation` and run the notebooks for the results you want to generate. Corresponding files will be generated in `/pretrain_comparison/output/results`

