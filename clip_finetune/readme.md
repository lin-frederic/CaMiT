# 🔧 CLIP Fine-Tuning & Evaluation Suite (LoRA-based)

This directory provides tools for **sequential fine-tuning** of a CLIP-based vision encoder using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685), along with scripts for evaluating model performance over time.

## 🧪 Fine-Tuning Script

### 🔹 `lora_clip_incremental.py`

- Implements **sequential LoRA fine-tuning** of a frozen CLIP encoder using **[OpenCLIP](https://github.com/mlfoundations/open_clip)**.
- Fine-tunes only low-rank adapter layers **without replay** of previous data.

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

---

## 📦 Requirements

Before running any scripts, make sure the following libraries are installed:

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- Standard Python libraries:
  - `torch`, `torchvision`
  - `numpy`, `tqdm`
  - `tensorboard`

