# Annotation Pipeline

This directory contains code for the **semi-automatic annotation** of fine-grained car models using large vision-language models, discriminative classifiers, and human validation.

## 📁 Folder Structure

### `qwen/`
- Uses **Qwen2.5-7B** (via HuggingFace `transformers`) to generate initial predictions for car crops.
- Outputs predicted car model and confidence score.

**Dependencies:**
- `transformers`
- `qwen-vl-utils`

---

### `chatgpt/`
- Uses **GPT-4o** (via OpenAI API) to verify or correct Qwen predictions.
- Employs focused prompting for better model recall and consistency.

**Dependencies:**
- `openai`
- `unidecode`
- `flask` (for web interface or local annotation tools)

---

### `students/`
- Trains supervised models (e.g., **DeiTQ**, **DeiTG**) using weak labels from Qwen/GPT.
- Models complement VLMs and boost prediction accuracy.

**Dependencies:**
- [`mocov3`](https://github.com/facebookresearch/moco-v3/tree/main)
- `timm`

---

### `clean_outlier/`
- Detects and removes annotation or embedding outliers to clean the dataset.
- Helps filter noisy labels before training.

**Dependencies:**
- [`SAM2`](https://github.com/facebookresearch/sam2)
- `flask`

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
- `flask`

---

## 🧰 Common Libraries

All modules also require standard Python libraries, including:

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `matplotlib`
- `tqdm`
