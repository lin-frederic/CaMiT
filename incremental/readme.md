# 📊 Incremental Evaluation Suite

This directory contains evaluation scripts for analyzing performance in **continual learning** scenarios with a focus on temporal data shifts.

## 🧪 Evaluation Methods

The following methods are implemented and used for assessing the quality and stability of learned visual representations over time:

### 🔹 `ncm.py`

- Standard **Nearest Class Mean (NCM)** classifier.

### 🔹 `ncm_v2.py`

- Implements **NCM-TI** (Time-Invariant NCM) as described in the paper.

### 🔹 `fecam_v3.py`

- Implements the **FeCAM** method used in the paper.

### 🔹 `fecam_common_v3.py`

- Variant of FeCAM that shares a **common covariance matrix** across all classes.

### 🔹 `ranpac.py`

- Implements the **RANPAC** method used in the paper.

### 🔹 `randumb.py`

- Implements the **RANDUMB** method used in the paper.

---

## 📦 Requirements


## 📦 Requirements

Before running any scripts, make sure the following libraries are installed:

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [MoCo v3](https://github.com/facebookresearch/moco-v3)
- [Transformers (Hugging Face)](https://github.com/huggingface/transformers)
- `qwen_vl_utils` (for Qwen-VL vision encoder support)
- Standard Python libraries:
  - `torch`, `torchvision`
  - `numpy`, `tqdm`
  - `tensorboard`
  - `scikit-learn`
