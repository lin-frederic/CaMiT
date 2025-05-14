## KID Computation
This script computes the Kernel Inception Distance (KID) between real and generated image embeddings for different models and scenarios.
## How to Run
```bash
python compute_kid.py MODEL_NAME --emb_root /path/to/embeddings
```

* ```MODEL_NAME```: The model variant whose embeddings will be loaded for KID calculation.
* ```--emb_root```: Root directory where embeddings are stored

## Example
```bash
python compute_kid.py clip_base --emb_root /home/data/bambaw/cars_finetune/embeddings
```
The script computes KID scores and saves results and plots in the current directory.
