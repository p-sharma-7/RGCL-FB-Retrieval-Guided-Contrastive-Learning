# RGCL: Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning
This is the official repo for the paper: 
- Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning (RGCL)
- Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection (LMM-RGCL)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-fine-tuning-of-large-multimodal-1/hateful-meme-classification-on-harm-p)](https://paperswithcode.com/sota/hateful-meme-classification-on-harm-p?p=improved-fine-tuning-of-large-multimodal-1)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-fine-tuning-of-large-multimodal/meme-classification-on-multioff)](https://paperswithcode.com/sota/meme-classification-on-multioff?p=improved-fine-tuning-of-large-multimodal-1)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-fine-tuning-of-large-multimodal/hateful-meme-classification-on-pridemm)](https://paperswithcode.com/sota/hateful-meme-classification-on-pridemm?p=improved-fine-tuning-of-large-multimodal-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-fine-tuning-of-large-multimodal/meme-classification-on-hateful-memes)](https://paperswithcode.com/sota/meme-classification-on-hateful-memes?p=improved-fine-tuning-of-large-multimodal-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improved-fine-tuning-of-large-multimodal/hateful-meme-classification-on-harmeme)](https://paperswithcode.com/sota/hateful-meme-classification-on-harmeme?p=improved-fine-tuning-of-large-multimodal-1)

- The link to the RGCL paper is [https://aclanthology.org/2024.acl-long.291.pdf](https://aclanthology.org/2024.acl-long.291.pdf).
- The link to the LMM-RGCL paper is [https://arxiv.org/abs/2502.13061](https://arxiv.org/abs/2502.13061).
- The link to the project page is [here](https://rgclmm.github.io/).


## Updates
- [18/02/2025] 🔥🔥🔥Our new work, LMM-RGCL, has been released. We further extend RGCL for large multimodal models. Check it out here: [https://arxiv.org/abs/2502.13061](https://arxiv.org/abs/2502.13061). We will shortly release checkpoints and code for LMM-RGCL.
- [29/10/2024] 🔥🔥Initial Release of the code base.
- [10/08/2024] 🔥Our paper appears at ACL2024 Main.

Useage
--------------------
## Create Env
```shell
conda create -n RGCL python=3.10 -y
conda activate RGCL
```

Install pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Install FAISS
```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
```

```
pip install -r requirements.txt
```


Dataset Preparation 
--------------------
#### Image data
Dump image into `./data/image/dataset_name/All` folder.
For example: `./data/image/FB/All/12345.png`, `./data/image/HarMeme/All`, `./data/image/Propaganda/All`, etc..
#### Annotation data
Dump `jsonl` annotation file into `./data/gt/dataset_name` folder.

#### Generate CLIP Embedding
We generate CLIP embedding prior to training to avoid repeated generation during training.

```shell
python3 src/utils/generate_CLIP_embedding_HF.py --dataset "FB"
python3 src/utils/generate_CLIP_embedding_HF.py --dataset "HarMeme"

```

#### Generate ALIGN Embedding
```shell
python3 src/utils/generate_ALIGN_embedding_HF.py --dataset "FB"
python3 src/utils/generate_ALIGN_embedding_HF.py --dataset "HarMeme"

```

#### Generate Sparse Retrieval Index
##### Generate VinVL Bounding Box Prediction (Optional)
We obtained the object detection bounding box with VinVL. To simplify your process to reproduce the results, we release the pre-extracted bbox prediction for the HatefulMemes dataset: [https://huggingface.co/datasets/Jingbiao/rgcl-sparse-retrieval/tree/main](https://huggingface.co/datasets/Jingbiao/rgcl-sparse-retrieval/tree/main)  


Training and Evalution 
--------------------
```
bash scripts\experiments.sh
```

## Common Issues
If you experience stuck in training, it might be due to the `faiss` installation. 

## Citation
If our work helped your research, please kindly cite our paper
```
@inproceedings{RGCL2024Mei,
    title = "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning",
    author = "Mei, Jingbiao  and
      Chen, Jinghong  and
      Lin, Weizhe  and
      Byrne, Bill  and
      Tomalin, Marcus",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.291",
    doi = "10.18653/v1/2024.acl-long.291",
    pages = "5333--5347"
}

@article{LMMRGCL2025Mei,
    title={Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection},
    url={http://arxiv.org/abs/2502.13061},
    DOI={10.48550/arXiv.2502.13061},
    note={arXiv:2502.13061 [cs]},
    number={arXiv:2502.13061},
    publisher={arXiv},
    author={Mei, Jingbiao and Chen, Jinghong and Yang, Guangyu and Lin, Weizhe and Byrne, Bill},
    year={2025},
    month=feb }


```
