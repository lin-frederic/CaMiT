# 🔧 MoCo v3 Fine-Tuning & Evaluation Suite

This directory contains scripts for fine-tuning a MoCo v3 Vision Transformer encoder using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685), as well as tools for various evaluation strategies.

## 🧪 Fine-Tuning Scripts

These scripts perform LoRA-based fine-tuning of a frozen MoCo v3 encoder pretrained on car crops.

### 🔹 `lora_mocov3_current.py`

- Fine-tunes a classifier using LoRA on a **single specified year**.
- Designed for isolated year-by-year adaptation.

### 🔹 `lora_mocov3_incremental.py`

- Performs **sequential LoRA fine-tuning**, incrementally updating the model as new years of data are introduced.
- No replay; past data is not revisited.

### 🔹 `lora_mocov3_replay.py`

- Implements **LoRA with rehearsal (replay)** by training on data from all **previous years + current**.
- Mitigates forgetting in time-evolving data.

---

## 📊 Evaluation Scripts

These scripts evaluate fine-tuned models using various strategies.

### 🔹 `ncm.py`, `ncm_v2.py`, `ncm_time.py`, `ncm_time_current.py`, `ncm_time_replay.py`

- Evaluate using **Nearest Class Mean (NCM)** classifiers.

### 🔹 `fecam.py`

- Evaluates using **FeCAM**

### 🔹 `ranpac.py`

- Evaluates using **RANPAC**

### 🔹 `randumb.py`

- Evaluates using **Randumb**.


## 📦 Requirements

Before running any scripts, make sure the following libraries are installed:

- [MoCo v3](https://github.com/facebookresearch/moco-v3)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- Standard Python libraries:
  - `torch`, `torchvision`
  - `timm`
  - `numpy`
  - `tqdm`
  - `tensorboard`