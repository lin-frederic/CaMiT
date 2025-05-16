# Annotation Pipeline

This directory contains code for the **semi-automatic annotation** of fine-grained car models using large vision-language models, discriminative classifiers, and human validation.

## 📁 Folder Structure

### `qwen/`
- Uses **Qwen2.5-7B** (decribed in [this report](https://arxiv.org/abs/2412.15115) and handled via HuggingFace [`transformers`](https://github.com/huggingface/transformers)) to generate initial predictions for car crops.
- Outputs predicted car model and confidence score.

**Dependencies:**
- [`transformers`](https://github.com/huggingface/transformers)
- [`qwen-vl-utils`](https://github.com/kq-chen/qwen-vl-utils)

---

### `chatgpt/`
- Uses **GPT-4o** (derived from [this report](https://arxiv.org/abs/2303.08774) and handled via OpenAI API) to verify or correct Qwen predictions.
- Employs focused prompting for better model recall and consistency.

**Dependencies:**
- [`openai`](https://github.com/openai/openai-python)
- [`unidecode`](https://github.com/takluyver/Unidecode)
- [`flask`](https://flask.palletsprojects.com/en/stable/) (for web interface or local annotation tools)

---

### `students/`
- Trains supervised models (e.g., [**DeiTQ**](https://github.com/facebookresearch/deit) or [**DeiTG**](https://github.com/facebookresearch/deit) form the [ICML 2021 paper](https://proceedings.mlr.press/v139/touvron21a.html)) using weak labels from Qwen/GPT.
- Models complement VLMs and boost prediction accuracy.

**Dependencies:**
- [`mocov3`](https://github.com/facebookresearch/moco-v3/tree/main)
- [`timm`](https://pypi.org/project/timm/)

---

### `clean_outlier/`
- Detects and removes annotation or embedding outliers to clean the dataset.
- Helps filter noisy labels before training.

**Dependencies:**
- [`SAM2`](https://github.com/facebookresearch/sam2) which implement the [ICLR 2025 paper](https://openreview.net/forum?id=Ha6RTeWMd0)
- [`flask`](https://flask.palletsprojects.com/en/stable/)

---

### `face_blurring/`
- Detects and blurs human faces in car images to ensure privacy.
  
**Dependencies:**
- [`insightface`](https://github.com/deepinsight/insightface)

---

### `manual_validation/`
- Interface for human annotators to validate predictions and resolve ambiguity.
- Supports majority voting across annotators.

**Dependencies:**
- [`flask`](https://flask.palletsprojects.com/en/stable/)

---

## 🧰 Common Libraries

All modules also require standard Python libraries, including:

- `torch` from [PyTorch](https://github.com/pytorch/pytorch)
- `torchvision` from [PyTorch](https://github.com/pytorch/pytorch)
- [`numpy`](https://github.com/numpy/numpy)
- [`opencv-python`](https://github.com/opencv/opencv)
- [`matplotlib`](https://github.com/matplotlib/matplotlib)
- [`tqdm`](https://github.com/tqdm/tqdm)
