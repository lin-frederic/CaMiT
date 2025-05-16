## README

### **CaMiT: A Time-Aware Car Model Dataset for Classification and Generation**

#### **Overview**
This repository contains the code and resources for the paper *"CaMiT: A Time-Aware Car Model Dataset for Classification and Generation"*. The dataset and associated code are designed to explore the temporal dynamics of car model representations and evaluate various time-aware learning strategies for fine-grained image classification and generation.

#### **Key Contributions**
1. **Dataset**: Introduces **CaMiT**, a large-scale, timestamped dataset of car models photographed over 17 years, contributed by 337K unique users.
2. **Experiments**: Evaluates the effectiveness of static pretraining (SPT), time-incremental pretraining (TIP), time-incremental classifier learning (TICL), and time-aware image generation (TAIG) on the CaMiT dataset.
3. **Findings**: Demonstrates the importance of time-aware approaches in mitigating temporal data shift and improving performance for both classification and generation tasks.

#### **Repository Structure**
The repository is organized into several folders, each corresponding to a specific component of the project. Below is a brief description of each folder:

---

### **Folder Structure**

#### **1. `annotation`**
-  Contains the code and scripts used for semi-automatic annotation of the CaMiT dataset.

#### **2. `clip_finetune`**
-  Code for finetuning CLIP models with LoRA.

#### **3. `incremental`**
-  Contains the implementation of time-incremental classifier learning (TICL) algorithms.

#### **4. `mocov3_finetune`**
-  Code for finetuning MoCo v3 models with LoRA for car model classification.

#### **5. `preprocessing`**
-  Scripts for detecting the cars from scraped images

#### **6. `pretrain`**
-  Code for pretraining MoCoV3 (TIP)

#### **7. `time_aware_generation`**
-  Implementation of time-aware image generation (TAIG) using Stable Diffusion.
---

#### **Dataset**
- The dataset is available (`here`)[https://huggingface.co/datasets/fredericlin/CaMiT]


#### **Running Experiments**
1. Pretrain Mocov3 models with code in `pretrain` or download (`checkpoints`)[https://huggingface.co/datasets/fredericlin/CaMiT-embeddings]
2. Navigate to the `Incremental` folder to run ncm evaluation scripts


#### **Evaluation**
- Results and metrics are reported in the paper. To reproduce the results, follow the instructions in the respective folder READMEs.