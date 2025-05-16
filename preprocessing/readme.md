# Preprocessing

This directory contains preprocessing scripts for preparing the Flickr Cars dataset. These scripts handle tasks like extracting data, detecting cars, cleaning, deduplication, and dataset organization.

---

## 📁 File Overview

| Script                | Description |
|-----------------------|-------------|
| `extract_tar.py`      | Extracts `.tar` files inside a specified folder. |
| `detect_cars.py`      | Runs car detection on images using a pretrained YOLOv11 detector. |
| `clear_corrupted.py`  | Removes corrupted or unreadable image files. |
| `deduplicate.py`      | Removes duplicate or near-duplicate images based on DINOv2 similarity. |
| `merge_detections.py` | Combines detection results from different runs or splits. |
| `datasets.py`         | Defines dataset structures used in deduplication. |
| `models.py`           | Defines models used in deduplication. |
| `utils.py`            | Common utility functions used across scripts. |

---

## ⚙️ Dependencies

- Python ≥ 3.9
- `torch` from [PyTorch](https://github.com/pytorch/pytorch)
- `torchvision` from [PyTorch](https://github.com/pytorch/pytorch)
- [`numpy`](https://github.com/numpy/numpy)
- [`Pillow`](https://github.com/python-pillow/Pillow)
- [`tqdm`](https://github.com/tqdm/tqdm)
- [`Ultralytics`](https://github.com/ultralytics/ultralytics) (for YOLOv11)

> **Install YOLOv11:**
> ```bash
> pip install git+https://github.com/ultralytics/ultralytics.git
> ```

---

## 🚀 Usage

Typical preprocessing workflow:

```bash
# 1. Extract image data
python extract_tar.py --tar_folder path/to/tarfolder --dest_folder path/to/images

# 2. Clean corrupted images
python clear_corrupted.py --data_path path/to/images

# 3. Run car detection
python detect_cars.py --data_path folder_containing_images --dest_path folder_containing_detections

# 4. Deduplicate
python deduplicate.py --data_path path/to/images --annotations_path path/to/car_detections --dest_path path/to/deduplication_results
