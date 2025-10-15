# RGCL-FB: Retrieval-Guided Contrastive Learning for Hateful Meme Detection (FB Dataset)

This repository contains a simplified, working implementation of:
> 📄 **[Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning (RGCL)](https://aclanthology.org/2024.acl-long.291.pdf)**  
> ACL 2024 · Jingbiao Mei et al.

---

## 🧠 What is RGCL?

**RGCL** improves multimodal hateful meme classification using **contrastive learning** and **retrieval-based augmentation**. It enhances traditional CLIP-based models by:
- Adding **ALIGN embeddings** alongside CLIP
- Introducing **hard negatives** for triplet loss
- Integrating **sparse textual retrieval** (object/attribute-based)

---

## ✅ What's Included in This Repo?

- ✔️ Code to preprocess and split image/text data
- ✔️ CLIP and ALIGN embedding generation scripts
- ✔️ Sparse retrieval generation using object/attribute text
- ✔️ RAC (Retrieval-Augmented Contrastive) model training
- ✔️ Inference script on **test_unseen** split
- ❌ No data, checkpoints, or wandb logs pushed
- ❌ RA-HMD & LLaMA-Factory components removed

---

## 📊 Dataset

This implementation uses the **Hateful Memes Expanded** dataset from Hugging Face:
- **Source**: [limjiayi/hateful_memes_expanded](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded/tree/main)
- **Description**: An expanded version of the Facebook Hateful Memes Challenge dataset with additional annotations and improved coverage
- **Format**: Images with corresponding text and binary labels (hateful/not hateful)

### Dataset Download
```bash
# Download dataset from Hugging Face
git clone https://huggingface.co/datasets/limjiayi/hateful_memes_expanded
# Or use the datasets library
pip install datasets
```

---

## ⚙️ Setup Instructions

### 1. Environment Setup

```bash
conda create -n RGCL python=3.10 -y
conda activate RGCL
```

Install dependencies:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install -r requirements.txt
```

### 2. 📁 Expected Data Structure

After placing files:
```
data/
├── image/FB/All/               # All original images
├── gt/FB/                      # All .jsonl annotation files
├── CLIP_Embedding/FB/          # Extracted CLIP features
├── ALIGN_Embedding/FB/         # Extracted ALIGN features
└── Sparse_Retrieval_Dict/FB/   # Generated retrieval dictionaries
```

---

## 📦 Data Preparation

### Step 1: Split images into train/dev/test folders
```bash
python src/utils/split_img.py
```

### Step 2: Generate Embeddings
```bash
python src/utils/generate_CLIP_embedding_HF.py --dataset "FB"
python src/utils/generate_ALIGN_embedding_HF.py --dataset "FB"
```

### Step 3: Generate Sparse Retrieval Index
```bash
python src/utils/generate_sparse_retrieval_dictionary.py --dataset "FB"
```

---

## 🏋️ Train the RGCL Model

Start training using:
```bash
bash scripts/experiments.sh
```

**Configuration:**
- Dataset: FB (Facebook Hateful Memes)
- Model: openai/clip-vit-large-patch14-336
- Fusion: ALIGN + CLIP + retrieval
- Loss: Triplet + hybrid
- Epochs: 30

---

## 🔍 Inference on test_unseen

After training:
```bash
python src/infer_test_unseen.py
```

This script:
- Loads best checkpoint from logging/
- Uses saved CLIP features of test_unseen
- Outputs predicted labels

---

## 📊 Evaluation Snapshot

| Split | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| dev_seen | 0.788 | 0.861 | 0.759 |
| test_seen | 0.750 | 0.849 | 0.711 |
| test_unseen | ✅ Tested via infer_test_unseen.py | | |

---

## 📌 Citation

If you use this repo or the original model, please cite:
```bibtex
@inproceedings{RGCL2024Mei,
    title = "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning",
    author = "Mei, Jingbiao and Chen, Jinghong and Lin, Weizhe and Byrne, Bill and Tomalin, Marcus",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
    year = "2024"
}
```

---

## 🙏 Acknowledgements

- Original authors: Jingbiao Mei et al.
- Official repo: github.com/JingbiaoMei/RGCL

---
